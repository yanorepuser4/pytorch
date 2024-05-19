""" Triton Implementation of the flex_attention Kernel"""

import logging
import math
from enum import auto, Enum
from typing import Any, List, Tuple

import torch
from torch._prims_common import make_contiguous_strides_for
from .. import config
from ..ir import (
    ComputedBuffer,
    FixedLayout,
    FlexibleLayout,
    InputBuffer,
    IRNode,
    StorageBox,
    Subgraph,
    TensorBox,
)
from ..lowering import empty_strided, full, lowerings, register_lowering
from ..select_algorithm import autotune_select_algorithm, TritonTemplate

log = logging.getLogger(__name__)
aten = torch.ops.aten


class SubgraphType(Enum):
    """The type of subgraph for which we want to generate an output buffer."""

    FWD = auto()  # Forward pass
    JOINT_FWD = auto()  # The recompute step fo the of the bwds kernel
    JOINT_BWD = auto()  # The bwd pass of the joint


def flex_attention_grid(batch_size, num_heads, num_queries, d_model, meta):
    """How is this kernel parallelized?
    We create a grid of (batch_size * num_heads, ceil_div(n_queries, query_block_size), 1)
    Each block is responsible for iterating over blocks of keys and values calculating
    the final attention output.
    """
    import triton

    return (triton.cdiv(num_queries, meta["BLOCK_M"]), batch_size * num_heads, 1)


def create_placeholder(
    name: str, dtype: torch.dtype, device: torch.device
) -> TensorBox:
    """Creates a placeholder input buffers for producing subgraph_output."""
    input_buffer = InputBuffer(name, FixedLayout(device, dtype, [1], [1]))
    return TensorBox.create(input_buffer)


def index_to_other_buffers(cnt: int, graph_type: SubgraphType) -> int:
    """This function needs to be aware of the signatures for flex_attention_forward
    and flex_attention_backward. If new args are added, or the signature changes
    be sure to update the indexing math

    Args:
        cnt (int): The current index of the placeholder node
        is_joint_graph (bool): Whether or not this subgraph represents the joint graph
    """
    # Current fwd_args = [query, key, value, score_mod, *other_buffers]
    # For fwd_graphs we have 5 dummy values this when the first lifted args
    # is seen cnt = 5 and the start of the index_buffers is at args[4]
    # thus we subtract 1 from the current cnt
    if graph_type == SubgraphType.FWD:
        return cnt - 1

    # Current bwd_args = [q, k, v, out, lse, grad_out, fw_graph, joint_graph, *other_buffers]
    # We have 5 dummy values but the start of other_buffers is at index 8
    if graph_type == SubgraphType.JOINT_FWD:
        return cnt + 3

    # Same bwd args but now with 6 dummy values while other_buffers still start at 8
    if graph_type == SubgraphType.JOINT_BWD:
        return cnt + 2


def build_subgraph_buffer(
    args: Tuple[IRNode],
    placeholder_inps: List[TensorBox],
    subgraph: Subgraph,
    graph_type: SubgraphType,
) -> ComputedBuffer:
    """This function's goal is to take in the required args and produce the subgraph buffer
    The subgraph buffer is a ComputedBuffer that will be inlined into the triton template

    Args:
        args: The args that were passed into the flex_attention kernel
        placeholder_inps: The list of scalar inputs, these were created on the fly through `create_placeholder`
        subgraph: The Subgraph ir for which to produce the output node
        graph_type: The type of subgraph for which we want to produce the output node, see enum above for details
    """
    cnt = 0
    env = {}
    for node in subgraph.graph_module.graph.nodes:
        # There are two classes of placeholder inpts that we need
        # to handle differently. For the first n_scalar_inps inputs
        # we expect that these placeholders were generated by the make_fx call
        # in the flex Attention HOP. So we need to create a new placeholder
        # TensorBox for each of these inputs. For the rest of the inputs we
        # expect that these are lifted inputs that fill up the '*other_buffers'
        # tuple and already have corresponding TensorBoxes passed in as args.
        if node.op == "placeholder":
            is_lifted_input = cnt >= len(placeholder_inps)
            lifted_input_index = index_to_other_buffers(cnt, graph_type)
            env[node] = (
                args[lifted_input_index] if is_lifted_input else placeholder_inps[cnt]
            )
            cnt += 1
        elif node.op == "call_function":
            # For call_function we use the default lowerings and pass in the
            # already created TensorBoxes as args
            from torch.utils._pytree import tree_map

            env[node] = lowerings[node.target](
                *tree_map(lambda x: env[x] if x in env else x, node.args)
            )
        elif node.op == "output":
            # For the output node we need to create a ComputedBuffer
            # which represents the actual score modification
            # The joint_graph's output should be of the form[grad_score, None, None, None, None]
            # This is because only the 'score' requires grad and the other outputs are
            # the non-differentiable index scalars
            if graph_type == SubgraphType.FWD or graph_type == SubgraphType.JOINT_FWD:
                output_node = node.args[0]
            else:
                output_node = node.args[0][0]
            output_buffer = env[output_node]
            assert isinstance(output_buffer, TensorBox), (
                "The output node  for flex attention's subgraph must be a TensorBox, but got: ",
                type(output_buffer),
            )
            assert isinstance(output_buffer.data, StorageBox), (
                "The output node for the flex attention subgraph must be a StorageBox, but got: ",
                type(output_buffer),
            )
            # Create the ComputedBuffer directly that will be inlined into the modification block
            subgraph_buffer = ComputedBuffer(
                name=None,
                layout=FlexibleLayout(
                    device=output_buffer.data.get_device(),
                    dtype=output_buffer.data.get_dtype(),
                    size=output_buffer.data.get_size(),
                ),
                data=output_buffer.data.data,  # type: ignore[arg-type]
            )
            return subgraph_buffer

    raise ValueError("TemplatedAttention was passed a subgraph with no output node!")


flex_attention_template = TritonTemplate(
    name="flex_attention",
    grid=flex_attention_grid,
    source=r"""
{{def_kernel("Q", "K", "V", "LSE")}}
    # Sub notation for this kernel:
    # Q: Query, K: Key, V: Value
    # M: Number of queries, N: Number of keys/values, D: Model dimension
    # z: Batch size, h: Number of heads, m: Number of queries per head, k: Number of keys per head
    # (Modifiable) Config options:
    # BLOCK_M
    # BLOCK_N
    # SCORE_MOD_IS_LINEAR: Is the score modifier linear? If so, we can lift the
    # change of base out of the loop
    # ROWS_GUARANTEED_SAFE: Is it guaranteed that at least one value in each row
    # is not masked out? If so, we can skip an extra safety check
    # OUTPUT_LOGSUMEXP: We only need to store the logsumexp if we require grad

    # Define Q Strides
    stride_qz = {{stride("Q", 0)}}
    stride_qh = {{stride("Q", 1)}}
    stride_qm = {{stride("Q", 2)}}
    stride_qk = {{stride("Q", 3)}}
    # Define K Strides
    stride_kz = {{stride("K", 0)}}
    stride_kh = {{stride("K", 1)}}
    stride_kn = {{stride("K", 2)}}
    stride_kk = {{stride("K", 3)}}
    # Define V Strides
    stride_vz = {{stride("V", 0)}}
    stride_vh = {{stride("V", 1)}}
    stride_vk = {{stride("V", 2)}}
    stride_vn = {{stride("V", 3)}}

    Z = {{size("Q", 0)}}
    H = {{size("Q", 1)}}
    M = {{size("Q", 2)}}
    N = {{size("K", 2)}}

    qk_scale = 1.0
    MATMUL_PRECISION = Q.dtype.element_ty

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    q_offset = off_hz * stride_qh
    kv_offset = off_hz * stride_kh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(M, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(BLOCK_DMODEL, N),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(Q_block_ptr)
    if SCORE_MOD_IS_LINEAR:
        qk_scale *= 1.44269504
    q = (q * qk_scale).to(MATMUL_PRECISION)
    # loop over k, v and update accumulator
    lo = 0
    hi = N
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k.to(MATMUL_PRECISION), acc=qk)
        # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
        m = offs_m[:, None]
        n = start_n + offs_n[None, :]
        {{ modification(
            subgraph_number=0,
            score="qk",
            b="off_hz // H",
            h="off_hz % H",
            m="m",
            n="n",
            out="qk"
        ) | indent_except_first(2) }}
        # TODO: In the case that score_mod is linear, this can be LICMed
        if not SCORE_MOD_IS_LINEAR:
            qk *= 1.44269504
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # -- compute scaling constant ---
        row_max = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, row_max)

        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        if not ROWS_GUARANTEED_SAFE:
            masked_out_rows = (m_i_new == float("-inf"))
            alpha = tl.where(masked_out_rows, 0, alpha)
            p = tl.where(masked_out_rows[:, None], 0, p)

        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc = tl.dot(p.to(MATMUL_PRECISION), v.to(MATMUL_PRECISION), acc)

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # Store output and logsumexp
    acc = acc / l_i[:, None]
    idx_z = tl.program_id(1) // H
    idx_h = tl.program_id(1) % H
    idx_m = offs_m[:, None]
    idx_d = tl.arange(0, BLOCK_DMODEL)[None, :]

    # TODO generalize and add proper mask support
    mask = (idx_m != -1) & (idx_d != -1)
    {{store_output(("idx_z", "idx_h", "idx_m", "idx_d"), "acc", "mask")}}

    # TODO dont want to write this if we dont require grad
    if OUTPUT_LOGSUMEXP:
        l_ptrs = LSE + off_hz * M + offs_m
        lse = m_i + tl.math.log2(l_i)
        tl.store(l_ptrs, lse)
 """,
)


_h100_default_config = {
    (torch.float32, 64): (128, 32, 4, 3),
    (torch.float32, 128): (32, 64, 4, 3),
    (torch.float32, 256): (32, 32, 4, 3),
    (torch.bfloat16, 64): (128, 64, 4, 3),
    (torch.bfloat16, 128): (64, 32, 4, 3),
    (torch.bfloat16, 256): (64, 32, 4, 3),
}

_a100_default_config = {
    (torch.float32, 64): (128, 32, 4, 3),
    (torch.float32, 128): (128, 32, 4, 3),
    (torch.float32, 256): (64, 16, 4, 3),
    (torch.bfloat16, 64): (128, 64, 4, 3),
    (torch.bfloat16, 128): (128, 32, 4, 3),
    (torch.bfloat16, 256): (32, 64, 4, 3),
}


def _get_default_config_fwd(query) -> Tuple[int, int, int, int]:
    dtype = query.get_dtype()
    head_dim = query.get_size()[-1]
    default_config = None

    if head_dim <= 256 and torch.cuda.get_device_capability() >= (9, 0):  # H100
        if dtype == torch.float32:
            default_config = (64, 64, 4, 3)
        else:
            default_config = (128, 64, 4, 3)
        default_config = _h100_default_config.get((dtype, head_dim), default_config)
    elif head_dim <= 256 and torch.cuda.get_device_capability() >= (8, 0):  # A100
        if dtype == torch.float32:
            default_config = (64, 64, 4, 3)
        else:
            default_config = (128, 64, 4, 3)
        default_config = _a100_default_config.get((dtype, head_dim), default_config)
    else:  # modest hardware or extremely large head_dim
        if dtype == torch.float32:
            default_config = (32, 16, 4, 3)
        else:
            default_config = (64, 32, 4, 3)

    return default_config


def _get_default_config_bwd(query) -> Tuple[int, int, int, int]:
    head_dim = query.get_size()[-1]
    dtype = query.get_dtype()

    if head_dim <= 256 and torch.cuda.get_device_capability() >= (9, 0):  # H100
        if dtype == torch.float32:
            return (64, 64, 4, 1)
        return (128, 128, 4, 3)
    elif head_dim <= 256 and torch.cuda.get_device_capability() >= (8, 0):  # A100
        return (32, 32, 4, 1)
    else:  # modest hardware or extremely large head_dim
        return (32, 32, 4, 1)


# TODO: We probably also need a layout constraint?
@register_lowering(torch.ops.higher_order.flex_attention, type_promotion_kind=None)
def flex_attention(*args, **kwargs):
    query, key, value, subgraph, *other_buffers = args
    placeholder_inps = [
        create_placeholder(name, dtype, query.get_device())
        for name, dtype in [
            ("score", query.get_dtype()),
            ("b", torch.int32),
            ("h", torch.int32),
            ("m", torch.int32),
            ("n", torch.int32),
        ]
    ]
    subgraph_buffer = build_subgraph_buffer(
        args, placeholder_inps, subgraph, graph_type=SubgraphType.FWD
    )
    layout = FixedLayout(
        query.get_device(),
        query.get_dtype(),
        query.get_size(),
        make_contiguous_strides_for(query.get_size()),
    )
    # see NOTE:[TritonTemplates with multiple outputs]
    logsumexp_shape = query.get_size()[:-1]  # [B, H, M]
    logsumexp = empty_strided(
        logsumexp_shape,
        None,
        dtype=torch.float32,  # The logsumexp is always stored in fp32 regardless of the input dtype
        device=query.get_device(),
    )
    choices: List[Any] = []
    configs: List[Tuple[int, int, int, int]] = []
    configs.append(_get_default_config_fwd(query))
    if config.max_autotune:
        configs += [
            (128, 64, 4, 3),
            (128, 128, 4, 3),
            (128, 128, 8, 2),
            (64, 128, 4, 3),
            (64, 64, 4, 3),
        ]

    # Note, we don't need to pass in the captured buffers explicitly
    # because they're implicitly added by the score_mod function
    # We do need to explicitly pass it in for autotuning though.
    for BLOCK_M, BLOCK_N, num_warps, num_stages in configs:
        flex_attention_template.maybe_append_choice(
            choices=choices,
            input_nodes=[query, key, value, logsumexp],
            layout=layout,
            subgraphs=[
                subgraph_buffer,
            ],
            mutated_inputs=[
                logsumexp,
            ],
            num_stages=num_stages,
            num_warps=num_warps,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=query.get_size()[-1],
            # For now, we always assume the "sound" option
            SCORE_MOD_IS_LINEAR=False,
            ROWS_GUARANTEED_SAFE=False,
            OUTPUT_LOGSUMEXP=True,
        )
    inputs_for_autotuning = [query, key, value, logsumexp] + list(other_buffers)
    return (
        autotune_select_algorithm(
            "flex_attention", choices, inputs_for_autotuning, layout
        ),
        logsumexp,
    )


# ---------------------------- Backward HOP Implementation ----------------------------


def flex_attention_backward_grid(batch_size, num_heads, num_key_value, d_model, meta):
    """How is this kernel parallelized?
    Currently this is only parallelizing over batch * num_heads, but we can, and want to
    parallelize over ceil_div(num_key_value, key_value_block_size). To do this will either require
    atomic updates to some grad values or to have a two pass kernel design.
    """
    return (batch_size * num_heads, 1, 1)


flex_attention_backward_template = TritonTemplate(
    name="flex_attention_backward",
    grid=flex_attention_backward_grid,
    source=r"""
{{def_kernel("Q", "K", "V", "OUT", "LSE", "DELTA", "DO", "DQ", "DV")}}
    # Sub notation for this kernel:
    # Q: Query, K: Key, V: Value
    # OUT: Forward output, LSE: logsumexp (logsumexp is always stored in fp32 regardless of the input dtype)
    # DELTA: Precomputed sum(OUT* DO, axis=1)
    # DO: Derivative of Output, DQ: Derivative of Query, DV: Derivative of Value
    # DK: Derivative of Key, is the written to via the store_output call due to some limitations with
    # inductor codegen
    # M: Number of queries, N: Number of keys/values, D: Model dimension
    # z: Batch size, h: Number of heads, m: Number of queries per head, k: Number of keys per head
    # (Modifiable) Config options:
    # BLOCK_M
    # BLOCK_N
    # SCORE_MOD_IS_LINEAR: Is the score modifier linear? If so, we can lift the
    # change of base out of the loop
    # ROWS_GUARANTEED_SAFE: Is it guaranteed that at least one value in each row
    # is not masked out? If so, we can skip an extra safety check
    # OUTPUT_LOGSUMEXP: We only need to store the logsumexp if we require grad

    # Define Q Strides
    stride_qz = {{stride("Q", 0)}}
    stride_qh = {{stride("Q", 1)}}
    stride_qm = {{stride("Q", 2)}}
    stride_qk = {{stride("Q", 3)}}
    # Define K Strides
    stride_kz = {{stride("K", 0)}}
    stride_kh = {{stride("K", 1)}}
    stride_kn = {{stride("K", 2)}}
    stride_kk = {{stride("K", 3)}}
    # Define V Strides
    stride_vz = {{stride("V", 0)}}
    stride_vh = {{stride("V", 1)}}
    stride_vn = {{stride("V", 2)}}
    stride_vk = {{stride("V", 3)}}

    Z = {{size("Q", 0)}}
    H = {{size("Q", 1)}}
    M = {{size("Q", 2)}}
    N = {{size("K", 2)}}

    qk_scale = 1.0
    MATMUL_PRECISION = Q.dtype.element_ty

    off_hz = tl.program_id(0)
    off_z = off_hz // H # batch idx
    off_h = off_hz % H # head idx

    # offset pointers for batch/head
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh

    # Asserting contiguous for now...
    DO += off_z * stride_qz + off_h * stride_qh
    DQ += off_z * stride_qz + off_h * stride_qh
    DV += off_z * stride_vz + off_h * stride_vh

    # TODO I think that this should be N_CTX/BLOCK_N blocks
    for start_n in range(0, NUM_Q_BLOCKS):
        # We are not doing the causal optimization yet allowing us to start further down the
        # kv column
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_DMODEL)

        # initialize pointers to value-like data
        q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
        do_ptrs = DO + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dq_ptrs = DQ + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)

        # pointer to row-wise quantities in value-like data
        D_ptrs = DELTA + off_hz * M
        l_ptrs = LSE + off_hz * N

        # initialize dv and dk
        dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

        # Key and Value stay in SRAM throughout
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        for start_m in range(0, NUM_Q_BLOCKS * BLOCK_M, BLOCK_M):
            offs_m_curr = start_m + offs_m

            # load q, k, v, do on-chip
            q = tl.load(q_ptrs)

            if SCORE_MOD_IS_LINEAR:
                qk_scale *= 1.44269504
            q = (q * qk_scale).to(MATMUL_PRECISION)

            # -- compute qk ---
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk = tl.dot(q, tl.trans(k.to(MATMUL_PRECISION)), acc=qk)
            pre_mod_scores = qk
            # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
            m = offs_m_curr[:, None]
            n = offs_n[None, :]
            {{ modification(
                subgraph_number=0,
                score="qk",
                b="off_z",
                h="off_h",
                m="m",
                n="n",
                out="qk"
            ) | indent_except_first(3) }}
            # TODO: In the case that score_mod is linear, this can be LICMed
            if not SCORE_MOD_IS_LINEAR:
                qk *= 1.44269504
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            l_i = tl.load(l_ptrs + offs_m_curr)
            p = tl.math.exp2(qk - l_i[:, None])

            # compute dv
            do = tl.load(do_ptrs)
            dv += tl.dot(tl.trans(p.to(MATMUL_PRECISION)), do)

            # compute dp = dot(v, do)
            Di = tl.load(D_ptrs + offs_m_curr) # [BLOCKM, 1]

            # compute ds = p * (dp - delta[:, None])
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do, tl.trans(v))
            ds = p * dp

            # ~~~~~~~~~~~~~~~~~~~ Apply joint modification  ~~~~~~~~~~~~~~~~~~~
            {{ modification(
                subgraph_number=1,
                score="pre_mod_scores",
                b="off_z",
                h="off_h",
                m="m",
                n="n",
                out="ds"
            ) | indent_except_first(3) }}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # compute dk = dot(ds.T, q)
            dk += tl.dot(tl.trans(ds.to(MATMUL_PRECISION)), q)
            # compute dq
            dq = tl.load(dq_ptrs)
            dq += tl.dot(ds.to(MATMUL_PRECISION), k)

            # Store grad_query
            tl.store(dq_ptrs, dq)

            # increment pointers
            dq_ptrs += BLOCK_M * stride_qm
            q_ptrs += BLOCK_M * stride_qm
            do_ptrs += BLOCK_M * stride_qm

        # write-back
        index_n = offs_n[:, None]
        index_k = offs_k[None, :]

        # Store grad_key and grad_value
        dv_ptrs = DV + (index_n * stride_vn + index_k * stride_vk)
        tl.store(dv_ptrs, dv)

        # TODO generalize and add proper mask support
        mask = (index_n != -1) & (index_k != -1)
        {{store_output(("off_z", "off_h", "index_n", "index_k"), "dk", "mask", indent_width=8)}}

 """,
)


# TODO: We probably also need a layout constraint?
@register_lowering(
    torch.ops.higher_order.flex_attention_backward, type_promotion_kind=None
)
def flex_attention_backward(*args, **kwargs):
    (
        query,
        key,
        value,
        out,
        logsumexp,
        grad_out,
        fw_graph,
        joint_graph,
        *other_buffers,
    ) = args

    device = query.get_device()
    dtype = query.get_dtype()

    fwd_placeholder_inps = [
        create_placeholder(name, dtype, device)
        for name, dtype in [
            ("score", dtype),
            ("b", torch.int32),
            ("h", torch.int32),
            ("m", torch.int32),
            ("n", torch.int32),
        ]
    ]
    fw_subgraph_buffer = build_subgraph_buffer(
        args, fwd_placeholder_inps, fw_graph, graph_type=SubgraphType.JOINT_FWD
    )

    joint_placeholder_inps = fwd_placeholder_inps + [
        create_placeholder("out", dtype, device)
    ]
    joint_subgraph_buffer = build_subgraph_buffer(
        args, joint_placeholder_inps, joint_graph, graph_type=SubgraphType.JOINT_BWD
    )

    layout_k = FixedLayout(
        key.get_device(),
        key.get_dtype(),
        key.get_size(),
        make_contiguous_strides_for(key.get_size()),
    )

    # Create delta which will is needed for the bwd's kernel
    mul_delta = lowerings[aten.mul](out, grad_out)
    delta = lowerings[aten.sum](mul_delta, axis=-1)

    # see NOTE:[TritonTemplates with multiple outputs]
    grad_query = full(
        query.get_size(), 0.0, dtype=dtype, device=device
    )  # torch.zeros equivalent
    grad_query.realize()
    grad_value = empty_strided(value.get_size(), None, dtype=dtype, device=device)

    choices: List[Any] = []
    configs: List[Tuple[int, int, int, int]] = []
    configs.append(_get_default_config_bwd(query))
    if config.max_autotune:
        configs += [
            (128, 128, 4, 3),
            (128, 128, 8, 1),
            (64, 64, 4, 3),
            (64, 64, 8, 1),
        ]

    for BLOCK_M, BLOCK_N, num_warps, num_stages in configs:
        flex_attention_backward_template.maybe_append_choice(
            choices=choices,
            input_nodes=[
                query,
                key,
                value,
                out,
                logsumexp,
                delta,
                grad_out,
                grad_query,
                grad_value,
            ],
            layout=layout_k,  # We use store_output only for grad_key
            subgraphs=[fw_subgraph_buffer, joint_subgraph_buffer],
            mutated_inputs=[grad_query, grad_value],
            num_stages=num_stages,
            num_warps=num_warps,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=query.get_size()[-1],
            NUM_Q_BLOCKS=math.ceil(query.get_size()[-2] / BLOCK_M),
            # For now, we always assume the "sound" option
            SCORE_MOD_IS_LINEAR=False,
        )
    inputs_for_autotuning = [
        query,
        key,
        value,
        out,
        logsumexp,
        delta,
        grad_out,
        grad_query,
        grad_value,
    ] + list(other_buffers)

    grad_key = autotune_select_algorithm(
        "flex_attention_backward", choices, inputs_for_autotuning, layout_k
    )
    return (
        grad_query,
        grad_key,
        grad_value,
    )
