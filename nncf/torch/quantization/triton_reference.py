# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_nmali/xa/cxajymrec2sunpfu77zr4xopfsic2z7ssp3bdqrjotbl22wliooo.py
# Topologically Sorted Source Nodes: [output, add, output_1, scale, output_2, neg, mul, zero_point, output_3, output_4, output_5], Original ATen: [aten.clamp, aten.add, aten.sub, aten.reciprocal, aten.mul, aten.neg, aten.round, aten.div]
# Source node to ATen node mapping:
#   add => add
#   mul => mul_1
#   neg => neg
#   output => clamp_max, clamp_min
#   output_1 => sub
#   output_2 => mul_2
#   output_3 => sub_1
#   output_4 => round_2
#   output_5 => div
#   scale => mul, reciprocal
#   zero_point => round_1
# Graph fragment:
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.Tensor](args = (%arg1_1, %arg2_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, %arg0_1), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.Tensor](args = (%clamp_min, %add), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_max, %arg2_1), kwargs = {})
#   %reciprocal : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%arg0_1,), kwargs = {})
#   %mul : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 255), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %mul), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%arg2_1,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, %mul), kwargs = {})
#   %round_1 : [num_users=1] = call_function[target=torch.ops.aten.round.default](args = (%mul_1,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_2, %round_1), kwargs = {})
#   %round_2 : [num_users=1] = call_function[target=torch.ops.aten.round.default](args = (%sub_1,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%round_2, %mul), kwargs = {})
triton_poi_fused_add_clamp_div_mul_neg_reciprocal_round_sub_0 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_neg_reciprocal_round_sub_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_neg_reciprocal_round_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '0E91B58DAB54C915AAF8467E3EDB6871F6D05685FF049BBEEDA70C789216121A', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_neg_reciprocal_round_sub_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262668288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 128256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = tmp1 + tmp3
    tmp5 = triton_helpers.minimum(tmp2, tmp4)
    tmp6 = tmp5 - tmp1
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp3
    tmp9 = 255.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp6 * tmp10
    tmp12 = -tmp1
    tmp13 = tmp12 * tmp10
    tmp14 = libdevice.nearbyint(tmp13)
    tmp15 = tmp11 - tmp14
    tmp16 = libdevice.nearbyint(tmp15)
    tmp17 = tmp16 / tmp10
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2048, 1), (1, 1))
    assert_size_stride(arg1_1, (2048, 128256), (128256, 1))
    assert_size_stride(arg2_1, (2048, 1), (1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2048, 128256), (128256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output, add, output_1, scale, output_2, neg, mul, zero_point, output_3, output_4, output_5], Original ATen: [aten.clamp, aten.add, aten.sub, aten.reciprocal, aten.mul, aten.neg, aten.round, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_neg_reciprocal_round_sub_0.run(arg1_1, arg2_1, arg0_1, buf0, 262668288, grid=grid(262668288), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
    return (buf0, )

def triton_forward(input_, input_low, input_range, levels):
    return call([input_low, input_, input_range])