import logging
import operator
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.export._trace

from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import (
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    TensorArgument,
)
from torch.fx import subgraph_rewriter
from torch.onnx.utils import _create_jit_graph

from torchgen.model import FunctionSchema


log = logging.getLogger(__name__)


def inplace_optimize_sym_size_div(gm: torch.fx.GraphModule):
    def pattern(im, dim, scale):
        sym_size_int = torch.ops.aten.sym_size.int(im, dim)
        scalar_tensor = torch.ops.aten.scalar_tensor(sym_size_int)
        div_scalar_mode = torch.ops.aten.div.Scalar_mode(
            scalar_tensor, scale, rounding_mode="trunc"
        )
        int_tensor = torch.ops.aten.Int.Tensor(div_scalar_mode)
        return int_tensor

    def replacement(im, dim, scale):
        sym_size_int = torch.ops.aten.sym_size.int(im, dim)
        return sym_size_int // scale

    replaced_patterns = subgraph_rewriter.replace_pattern(gm, pattern, replacement)


def normalize_name(name: str) -> str:
    return name.replace(".", "_")


def get_op_overload(node: torch._C.Node):
    schema_str = node.schema()
    schema = FunctionSchema.parse(schema_str)
    ns, op_name = str(schema.name.name).split("::")
    override = schema.name.overload_name

    op_overload_packet = getattr(torch.ops.aten, op_name)
    if override:
        op_overload = getattr(op_overload_packet, override)
    else:
        op_overload = op_overload_packet.default

    return op_overload


class TS2FXGraphConverter:
    def __init__(
        self,
        ts_graph: Union[torch._C.Graph, torch._C.Block],
        param_names: Set[str],
        buffer_names: Set[str],
    ):
        self.ts_graph = ts_graph
        self.param_names = param_names
        self.buffer_names = buffer_names

        self.fx_graph: torch.fx.Graph = torch.fx.Graph()
        self.input_specs: List[InputSpec] = []
        self.output_specs: List[OutputSpec] = []

        self.name_to_node: Dict[str, Union[torch.fx.Node, List[torch.fx.Node]]] = {}
        self.constant_map: Dict[str, Any] = {}
        self.attribute_map: Dict[str, Any] = {}
        self.tensor_constants: Dict[str, torch.Tensor] = {}

        self.subgraphs: Dict[str, torch.fx.GraphModule] = {}

    def add_subgraph(self, subgraph) -> str:
        name = f"subgraph_{len(self.subgraphs)}"
        self.subgraphs[name] = subgraph
        return name

    def get_args_kwargs(self, node: torch._C.Node, schema):
        args = []
        kwargs = {}
        for input, schema_arg in zip(node.inputs(), schema.arguments):
            if schema_arg.kwarg_only:
                kwargs[schema_arg.name] = self.get_fx_value(input)
            else:
                args.append(self.get_fx_value(input))

        return tuple(args), kwargs

    def get_fx_value(self, value: torch._C.Value):
        value_name = value.debugName()
        if value_name in self.name_to_node:
            input_node = self.name_to_node[value_name]
            return input_node
        elif value_name in self.attribute_map:
            attr_name = self.attribute_map[value_name]
            if attr_name in self.name_to_node:
                input_node = self.name_to_node[attr_name]
                return input_node
            else:
                raise ValueError(f"Value {attr_name} not found")
        elif value_name in self.constant_map:
            return self.constant_map[value_name]
        else:
            raise ValueError(f"Input {value_name} not found")

    def convert(self) -> torch.fx.GraphModule:
        self.convert_graph_inputs()

        for node in self.ts_graph.nodes():
            self.convert_node(node)

        self.convert_graph_outputs()

        gm = torch.fx.GraphModule(self.subgraphs, self.fx_graph)

        inplace_optimize_sym_size_div(gm)

        gm.graph.lint()

        return gm

    def convert_graph_inputs(self):
        for graph_input in self.ts_graph.inputs():
            name = graph_input.debugName()
            normalized_name = normalize_name(name)

            fx_node = self.fx_graph.placeholder(normalized_name)

            # fx_node.meta["val"] = FakeTensor()
            # TODO: set fx_node.meta["val"]

            self.name_to_node[name] = fx_node

            if name in self.param_names:
                self.input_specs.append(
                    InputSpec(
                        InputKind.PARAMETER,
                        arg=TensorArgument(name=normalized_name),
                        target=name,
                    )
                )
            elif name in self.buffer_names:
                self.input_specs.append(
                    InputSpec(
                        InputKind.BUFFER,
                        arg=TensorArgument(name=normalized_name),
                        target=name,
                        persistent=True,
                    )
                )
            else:
                self.input_specs.append(
                    InputSpec(
                        InputKind.USER_INPUT,
                        arg=TensorArgument(name=normalized_name),
                        target=name,
                    )
                )

    def convert_prim_Constant(self, node: torch._C.Node):
        name = node.output().debugName()

        value: Any = None
        if node.hasAttribute("value"):
            constant_kind = node.kindOf("value")
            if constant_kind == "i":
                value = node.i("value")
            elif constant_kind == "f":
                value = node.f("value")
            elif constant_kind == "s":
                value = node.s("value")
            elif constant_kind == "t":
                # lift tensor constant as a placeholder
                placeholder_name = f"constant_{name}"
                fx_node = self.fx_graph.placeholder(placeholder_name)
                self.name_to_node[name] = fx_node
                self.tensor_constants[placeholder_name] = node.t("value")

                self.input_specs.append(
                    InputSpec(
                        InputKind.CONSTANT_TENSOR,
                        arg=TensorArgument(name=placeholder_name),
                        target=placeholder_name,
                    )
                )

                value = fx_node
            elif constant_kind == "ival":
                value = node.ival("value")
            else:
                raise ValueError(f"Unsupported constant type: {node.kindOf('value')}")
        else:
            value = None

        self.constant_map[name] = value

    def convert_prim_GetAttr(self, node: torch._C.Node):
        def get_attr(name: str):
            if name in self.attribute_map:
                return self.attribute_map[name]
            else:
                raise ValueError(f"Attribute {name} not found")

        output_name = node.output().debugName()

        attr_name = node.s("name")
        input_name = node.input().debugName()

        root_attr_name = get_attr(input_name)
        self.attribute_map[output_name] = (
            f"{root_attr_name}.{attr_name}" if root_attr_name else attr_name
        )

    def convert_aten_op(self, node: torch._C.Node):
        try:
            target = get_op_overload(node)
        except Exception as e:
            raise RuntimeError(f"Unsupported node {node.kind()}") from e

        if target is torch.ops.aten.size.int:
            target = torch.ops.aten.sym_size.int

        args, kwargs = self.get_args_kwargs(node, target._schema)

        fx_node = self.fx_graph.call_function(target, args, kwargs)

        # TODO: covnert sourceRange() into stack_trace
        # fx_node.meta["stack_trace"] = node.sourceRange()

        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_prim_ListConstruct(self, node: torch._C.Node):
        output_list = []
        for input in node.inputs():
            output_list.append(self.get_fx_value(input))

        output_name = node.output().debugName()
        self.name_to_node[output_name] = output_list

    def convert_prim_TupleIndex(self, node: torch._C.Node):
        args = tuple(self.get_fx_value(input) for input in node.inputs())
        getitem_node = self.fx_graph.call_function(operator.getitem, args)

        output_name = node.output().debugName()
        self.name_to_node[output_name] = getitem_node

    def convert_aten_Int(self, node: torch._C.Node):
        # converts aten::Int as aten._to_copy + aten::_local_scalar_dense
        target = torch.ops.aten._to_copy.default
        args = tuple(self.get_fx_value(input) for input in node.inputs())
        to_copy_node = self.fx_graph.call_function(target, args, {"dtype": torch.int32})

        fx_node = self.fx_graph.call_function(
            torch.ops.aten._local_scalar_dense.default, (to_copy_node,)
        )

        # TODO: covnert sourceRange() into stack_trace
        # fx_node.meta["stack_trace"] = node.sourceRange()

        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_prim_NumToTensor(self, node: torch._C.Node):
        # converts prim::NumToTensor as aten.scalar_tensor
        target = torch.ops.aten.scalar_tensor
        args = tuple(self.get_fx_value(input) for input in node.inputs())

        fx_node = self.fx_graph.call_function(target, args)

        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_prim_CreateObject(self, node: torch._C.Node):
        output_name = node.output().debugName()
        self.attribute_map[output_name] = ""

    def convert_aten__convolution(self, node: torch._C.Node):
        # converts aten::_convolution as aten.convolution, since aten::_convolution
        # doesn't have a meta function
        target = torch.ops.aten.convolution.default
        args, kwargs = self.get_args_kwargs(node, target._schema)

        fx_node = self.fx_graph.call_function(target, args, kwargs)

        output_name = node.output().debugName()
        self.name_to_node[output_name] = fx_node

    def convert_aten_div(self, node: torch._C.Node):
        target = get_op_overload(node)
        schema = target._schema

        args, kwargs = self.get_args_kwargs(node, schema)

        # converts aten::div.Tensor_mode(x, tensor_constant)
        # as aten.div.Scalar_mode(x, tensor_constant.item())
        if schema.overload_name == "Tensor_mode":
            arg1_name = args[1].name
            if arg1_name in self.tensor_constants:
                tensor_constant = self.tensor_constants[arg1_name]
                if tensor_constant.numel() == 1:
                    updated_args = list(args)
                    updated_args[1] = self.tensor_constants[arg1_name].item()

                    fx_node = self.fx_graph.call_function(
                        torch.ops.aten.div.Scalar_mode,
                        tuple(updated_args),
                        kwargs,
                    )

                    # TODO: covnert sourceRange() into stack_trace
                    # fx_node.meta["stack_trace"] = node.sourceRange()

                    output_name = node.output().debugName()
                    self.name_to_node[output_name] = fx_node
                    return

        self.convert_aten_op(node)

    def convert_prim_if(self, node: torch._C.Node):
        inputs = list(node.inputs())
        assert len(inputs) == 1
        predicate = self.get_fx_value(inputs[0])

        # Get union of inputs to blocks
        arguments = set()
        for block in node.blocks():
            block_args = set()

            # TODO: block.inputs(), not sure what theyre used for

            for block_node in block.nodes():
                for block_node_in in block_node.inputs():
                    if block_node_in.debugName() in self.name_to_node:
                        block_args.add(block_node_in.debugName())

            arguments.update(block_args)

        arguments = list(arguments)

        # Convert blocks to subgraphs
        subgraph_nodes = []
        for block in node.blocks():
            subgraph_converter = TS2FXGraphConverter(block, set(), set())
            subgraph_converter.constant_map = self.constant_map

            for block_arg in arguments:
                normalized_block_arg_name = normalize_name(block_arg)
                placeholder_node = subgraph_converter.fx_graph.placeholder(
                    normalized_block_arg_name
                )
                subgraph_converter.name_to_node[block_arg] = placeholder_node

            subgraph = subgraph_converter.convert()
            subgraph_name = self.add_subgraph(subgraph)
            subgraph_nodes.append(self.fx_graph.get_attr(subgraph_name))

        assert len(subgraph_nodes) == 2

        fx_block_args = [self.name_to_node[arg_name] for arg_name in arguments]
        args = (
            predicate,
            subgraph_nodes[0],
            subgraph_nodes[1],
            tuple(fx_block_args),
        )

        cond_node = self.fx_graph.call_function(torch.cond, args, {})

        output_name = node.output().debugName()
        self.name_to_node[output_name] = cond_node

    def convert_as_noop(self, node: torch._C.Node):
        # Converts the node as a no-op by mapping its output node as arg[0]

        target = get_op_overload(node)
        schema = target._schema

        args, kwargs = self.get_args_kwargs(node, schema)

        output_name = node.output().debugName()
        self.name_to_node[output_name] = args[0]

    def convert_node(self, node: torch._C.Node):
        node_kind = node.kind()
        if node_kind == "prim::CreateObject":
            self.convert_prim_CreateObject(node)
        elif node_kind == "prim::Constant":
            self.convert_prim_Constant(node)
        elif node_kind == "prim::GetAttr":
            self.convert_prim_GetAttr(node)
        elif node_kind == "prim::NumToTensor":
            self.convert_prim_NumToTensor(node)
        elif node_kind in ["prim::ListConstruct", "prim::TupleConstruct"]:
            self.convert_prim_ListConstruct(node)
        elif node_kind == "prim::TupleIndex":
            self.convert_prim_TupleIndex(node)
        # elif node_kind == "aten::Int":
        #     convert_aten_Int(node)
        elif node_kind == "aten::_convolution":
            self.convert_aten__convolution(node)
        elif node_kind == "aten::div":
            self.convert_aten_div(node)
        elif node_kind == "prim::If":
            self.convert_prim_if(node)
        elif node_kind == "aten::Bool":
            self.convert_as_noop(node)
        elif node_kind.startswith("aten::"):
            self.convert_aten_op(node)
        else:
            raise ValueError(f"Unsupported node kind: {node_kind}")

    def convert_graph_outputs(self):
        args = []
        for graph_output in self.ts_graph.outputs():
            output_name = graph_output.debugName()
            if output_name in self.name_to_node:
                args.append(self.name_to_node[output_name])
            else:
                raise ValueError(f"Output {output_name} not found")

            self.output_specs.append(
                OutputSpec(
                    OutputKind.USER_OUTPUT,
                    arg=TensorArgument(name=output_name),
                    target=output_name,
                )
            )

        self.fx_graph.output(args[0])


class TS2EPConverter:
    # TorchScript model to ExportedProgram converter
    def __init__(
        self,
        ts_model,
        sample_args: Tuple[Any, ...],
        sample_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.ts_model = ts_model
        self.ts_graph, self.params, _, _ = _create_jit_graph(ts_model, sample_args)

        self.sample_args = sample_args
        self.sample_kwargs = sample_kwargs

        self.param_names: Set[str] = {name for name, _ in ts_model.named_parameters()}
        self.buffer_names: Set[str] = {name for name, _ in ts_model.named_buffers()}

    def convert(self) -> ExportedProgram:
        graph_converter = TS2FXGraphConverter(
            self.ts_graph, self.param_names, self.buffer_names
        )
        gm = graph_converter.convert()
        ep = self.retrace_as_exported_program(gm, graph_converter.tensor_constants)
        return ep

    def retrace_as_exported_program(self, gm: torch.fx.GraphModule, tensor_constants):
        # TODO: adjust input orders to match GraphSignature convention
        inputs = [*self.sample_args, *self.params, *tensor_constants.values()]

        ep = torch.export._trace._export(
            gm,
            tuple(inputs),
            strict=False,
            pre_dispatch=True,
        )
        return ep
