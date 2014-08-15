;;  Copyright (c) 2013, Intel Corporation
;;  All rights reserved.
;;
;;  Redistribution and use in source and binary forms, with or without
;;  modification, are permitted provided that the following conditions are
;;  met:
;;
;;    * Redistributions of source code must retain the above copyright
;;      notice, this list of conditions and the following disclaimer.
;;
;;    * Redistributions in binary form must reproduce the above copyright
;;      notice, this list of conditions and the following disclaimer in the
;;      documentation and/or other materials provided with the distribution.
;;
;;    * Neither the name of Intel Corporation nor the names of its
;;      contributors may be used to endorse or promote products derived from
;;      this software without specific prior written permission.
;;
;;
;;   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
;;   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
;;   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
;;   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
;;   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
;;   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
;;   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
;;   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
;;   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
;;   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
;;   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  

;; TODO
;; - Use vector ctlz, cttz for varying versions of these in stdlib--currently,
;;   these dispatch out to do one lane at a time.  There are LLVM intrinsics
;;   for these now, so can we just use those for everything?

define(`WIDTH',`16')
define(`MASK',`i1')
define(`HAVE_GATHER',`1')
define(`HAVE_SCATTER',`1')

include(`util.m4')

stdlib_core()
packed_load_and_store()
scans()
int64minmax()

ctlztz()
define_prefetches()
define_shuffles()
aossoa()

rdrand_definition()

trigonometry_decl()
transcendetals_decl()

saturation_arithmetic_vec16()

;; svml
include(`svml.m4')
svml_declare(float,f16,16)
svml_define(float,f16,16,f)
svml_declare(double,8,8)
svml_define_x(double,8,8,d,16)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp

declare float @__rcp_uniform_float(float) nounwind readonly alwaysinline

declare  double @__rcp_uniform_double(double)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding floats

declare float @__round_uniform_float(float) nounwind readonly alwaysinline
declare float @__floor_uniform_float(float) nounwind readonly alwaysinline
declare float @__ceil_uniform_float(float) nounwind readonly alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding doubles

declare double @__round_uniform_double(double) nounwind readonly alwaysinline
declare double @__floor_uniform_double(double) nounwind readonly alwaysinline
declare double @__ceil_uniform_double(double) nounwind readonly alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rsqrt

declare float @__rsqrt_uniform_float(float) nounwind readonly alwaysinline
declare double @__rsqrt_uniform_double(double) nounwind readonly alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; sqrt

declare float @sqrtf(float) readnone

define float @__sqrt_uniform_float(float) nounwind readonly alwaysinline {
  %v = call float @sqrtf(float %0)
  ret float %v
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; fastmath

;; TODO: is there an avx512 (non-sse) variant?
declare void @llvm.x86.sse.stmxcsr(i8 *) nounwind
declare void @llvm.x86.sse.ldmxcsr(i8 *) nounwind

define void @__fastmath() nounwind alwaysinline {
  %ptr = alloca i32
  %ptr8 = bitcast i32 * %ptr to i8 *
  call void @llvm.x86.sse.stmxcsr(i8 * %ptr8)
  %oldval = load i32 *%ptr

  ; turn on DAZ (64)/FTZ (32768) -> 32832
  %update = or i32 %oldval, 32832
  store i32 %update, i32 *%ptr
  call void @llvm.x86.sse.ldmxcsr(i8 * %ptr8)
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float min/max

declare float @__max_uniform_float(float, float) nounwind readonly alwaysinline
declare float @__min_uniform_float(float, float) nounwind readonly alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int min/max

declare i32 @__min_uniform_int32(i32, i32) nounwind readonly alwaysinline
declare i32 @__max_uniform_int32(i32, i32) nounwind readonly alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unsigned int min/max

declare i32 @__min_uniform_uint32(i32, i32) nounwind readonly alwaysinline
declare i32 @__max_uniform_uint32(i32, i32) nounwind readonly alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; horizontal ops

declare i32 @llvm.ctpop.i32(i32) nounwind readnone

define i32 @__popcnt_int32(i32) nounwind readonly alwaysinline {
  %call = call i32 @llvm.ctpop.i32(i32 %0)
  ret i32 %call
}

declare i64 @llvm.ctpop.i64(i64) nounwind readnone

define i64 @__popcnt_int64(i64) nounwind readonly alwaysinline {
  %call = call i64 @llvm.ctpop.i64(i64 %0)
  ret i64 %call
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision sqrt

declare double @__sqrt_uniform_double(double) nounwind alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision min/max

declare double @__min_uniform_double(double, double) nounwind readnone alwaysinline
declare double @__max_uniform_double(double, double) nounwind readnone alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int8/int16 builtins

define_avgs()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rcp

declare <16 x float> @__rcp_varying_float(<16 x float>) nounwind readonly alwaysinline

declare <16 x double> @__rcp_varying_double(<16 x double>) nounwind readonly alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding floats

declare <16 x float> @__round_varying_float(<16 x float>) nounwind readonly alwaysinline
declare <16 x float> @__floor_varying_float(<16 x float>) nounwind readonly alwaysinline
declare <16 x float> @__ceil_varying_float(<16 x float>) nounwind readonly alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding doubles

declare <16 x double> @__round_varying_double(<16 x double>) nounwind readonly alwaysinline
declare <16 x double> @__floor_varying_double(<16 x double>) nounwind readonly alwaysinline
declare <16 x double> @__ceil_varying_double(<16 x double>) nounwind readonly alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rsqrt

declare <16 x float> @__rsqrt_varying_float(<16 x float> %v) nounwind readonly alwaysinline
declare <16 x double> @__rsqrt_varying_double(<16 x double> %v) nounwind readonly alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; sqrt

declare <16 x float> @llvm.sqrt.v16f32(<16 x float>)

define <16 x float> @__sqrt_varying_float(<16 x float>) nounwind readonly alwaysinline {
  %r = call <16 x float> @llvm.sqrt.v16f32(<16 x float> %0)
  ret <16 x float> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float min/max

declare <16 x float> @__max_varying_float(<16 x float>,
                                          <16 x float>) nounwind readonly alwaysinline
declare <16 x float> @__min_varying_float(<16 x float>,
                                          <16 x float>) nounwind readonly alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; horizontal ops

declare i64 @__movmsk(<16 x MASK>) nounwind readnone alwaysinline
declare i1 @__any(<16 x MASK>) nounwind readnone alwaysinline
declare i1 @__all(<16 x MASK>) nounwind readnone alwaysinline
declare i1 @__none(<16 x MASK>) nounwind readnone alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal float ops

declare float @__reduce_add_float(<16 x float>) nounwind readonly alwaysinline
declare float @__reduce_min_float(<16 x float>) nounwind readnone alwaysinline
declare float @__reduce_max_float(<16 x float>) nounwind readnone alwaysinline

reduce_equal(16)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal int8 ops

declare i16 @__reduce_add_int8(<16 x i8>) nounwind readnone alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal int16 ops

define internal <16 x i16> @__add_varying_i16(<16 x i16>,
                                  <16 x i16>) nounwind readnone alwaysinline {
  %r = add <16 x i16> %0, %1
  ret <16 x i16> %r
}

define internal i16 @__add_uniform_i16(i16, i16) nounwind readnone alwaysinline {
  %r = add i16 %0, %1
  ret i16 %r
}

define i16 @__reduce_add_int16(<16 x i16>) nounwind readnone alwaysinline {
  reduce16(i16, @__add_varying_i16, @__add_uniform_i16)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal int32 ops

define <16 x i32> @__add_varying_int32(<16 x i32>,
                                       <16 x i32>) nounwind readnone alwaysinline {
  %s = add <16 x i32> %0, %1
  ret <16 x i32> %s
}

define i32 @__add_uniform_int32(i32, i32) nounwind readnone alwaysinline {
  %s = add i32 %0, %1
  ret i32 %s
}

define i32 @__reduce_add_int32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__add_varying_int32, @__add_uniform_int32)
}


define i32 @__reduce_min_int32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__min_varying_int32, @__min_uniform_int32)
}


define i32 @__reduce_max_int32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__max_varying_int32, @__max_uniform_int32)
}

define i32 @__reduce_min_uint32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__min_varying_uint32, @__min_uniform_uint32)
}

define i32 @__reduce_max_uint32(<16 x i32>) nounwind readnone alwaysinline {
  reduce16(i32, @__max_varying_uint32, @__max_uniform_uint32)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal double ops

declare double @__reduce_add_double(<16 x double>) nounwind readonly alwaysinline

define double @__reduce_min_double(<16 x double>) nounwind readnone alwaysinline {
  reduce16(double, @__min_varying_double, @__min_uniform_double)
}

define double @__reduce_max_double(<16 x double>) nounwind readnone alwaysinline {
  reduce16(double, @__max_varying_double, @__max_uniform_double)
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; horizontal int64 ops

define <16 x i64> @__add_varying_int64(<16 x i64>,
                                       <16 x i64>) nounwind readnone alwaysinline {
  %s = add <16 x i64> %0, %1
  ret <16 x i64> %s
}

define i64 @__add_uniform_int64(i64, i64) nounwind readnone alwaysinline {
  %s = add i64 %0, %1
  ret i64 %s
}

define i64 @__reduce_add_int64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__add_varying_int64, @__add_uniform_int64)
}


define i64 @__reduce_min_int64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__min_varying_int64, @__min_uniform_int64)
}


define i64 @__reduce_max_int64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__max_varying_int64, @__max_uniform_int64)
}


define i64 @__reduce_min_uint64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__min_varying_uint64, @__min_uniform_uint64)
}


define i64 @__reduce_max_uint64(<16 x i64>) nounwind readnone alwaysinline {
  reduce16(i64, @__max_varying_uint64, @__max_uniform_uint64)
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unaligned loads/loads+broadcasts

; no masked load instruction for i8 and i16 types??
masked_load(i8,  1)
masked_load(i16, 2)

declare <16 x i32> @__masked_load_i32(i8 *, <16 x MASK> %mask) nounwind alwaysinline
declare <16 x i64> @__masked_load_i64(i8 *, <16 x MASK> %mask) nounwind alwaysinline

masked_load_float_double()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; masked store

declare void @__masked_store_i8(<16 x i8>* nocapture %ptr, <16 x i8> %val, 
                                <16 x MASK> %mask) nounwind alwaysinline
declare void @__masked_store_i16(<16 x i16>* nocapture %ptr, <16 x i16> %val,
                                 <16 x MASK> %mask) nounwind alwaysinline
declare void @__masked_store_i32(<16 x i32>* nocapture %ptr, <16 x i32> %val, 
                                 <16 x MASK> %mask) nounwind alwaysinline
declare void @__masked_store_i64(<16 x i64>* nocapture %ptr, <16 x i64> %val,
                                 <16 x MASK> %mask) nounwind alwaysinline


define void @__masked_store_blend_i8(<16 x i8>* nocapture %ptr, <16 x i8> %new, 
                                     <16 x MASK> %mask) nounwind alwaysinline {
  %old = load <16 x i8>* %ptr
  %sel = select <16 x i1> %mask, <16 x i8> %new, <16 x i8> %old
  store <16 x i8> %sel, <16 x i8>* %ptr
  ret void
}

define void @__masked_store_blend_i16(<16 x i16>* nocapture %ptr, <16 x i16> %new, 
                                 <16 x MASK> %mask) nounwind alwaysinline {
  %old = load <16 x i16>* %ptr
  %sel = select <16 x i1> %mask, <16 x i16> %new, <16 x i16> %old
  store <16 x i16> %sel, <16 x i16>* %ptr
  ret void
}

define void @__masked_store_blend_i32(<16 x i32>* nocapture %ptr, <16 x i32> %new, 
                                      <16 x MASK> %mask) nounwind alwaysinline {
  %old = load <16 x i32>* %ptr
  %sel = select <16 x i1> %mask, <16 x i32> %new, <16 x i32> %old
  store <16 x i32> %sel, <16 x i32>* %ptr
  ret void
}

define void @__masked_store_blend_i64(<16 x i64>* nocapture %ptr, <16 x i64> %new, 
                                      <16 x MASK> %mask) nounwind alwaysinline {
  %old = load <16 x i64>* %ptr
  %sel = select <16 x i1> %mask, <16 x i64> %new, <16 x i64> %old
  store <16 x i64> %sel, <16 x i64>* %ptr
  ret void
}

masked_store_float_double()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; scatter

define(`scatter_avx512_tmp', `
declare void @__scatter_base_offsets32_$1(i8* nocapture, i32, <WIDTH x i32>,
                                          <WIDTH x $1>, <WIDTH x i1>) nounwind 
declare void @__scatter_base_offsets64_$1(i8* nocapture, i32, <WIDTH x i64>,
                                          <WIDTH x $1>, <WIDTH x i1>) nounwind 
declare void @__scatter32_$1(<WIDTH x i32>, <WIDTH x $1>,
                             <WIDTH x i1>) nounwind 
declare void @__scatter64_$1(<WIDTH x i64>, <WIDTH x $1>,
                              <WIDTH x i1>) nounwind 
')

scatter_avx512_tmp(i8)
scatter_avx512_tmp(i16)
scatter_avx512_tmp(i32)
scatter_avx512_tmp(float)
scatter_avx512_tmp(i64)
scatter_avx512_tmp(double)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision sqrt

declare <8 x double> @llvm.sqrt.v8f64(<8 x double>)

define <16 x double> @__sqrt_varying_double(<16 x double>) nounwind alwaysinline {
  unary8to16(r, double, @llvm.sqrt.v8f64, %0)
  ret <16 x double> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision min/max

declare <16 x double> @__min_varying_double(<16 x double>, <16 x double>) nounwind readnone alwaysinline

declare <16 x double> @__max_varying_double(<16 x double>, <16 x double>) nounwind readnone alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int min/max

declare <16 x i32> @__min_varying_int32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline
declare <16 x i32> @__max_varying_int32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unsigned int min/max

declare <16 x i32> @__min_varying_uint32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline
declare <16 x i32> @__max_varying_uint32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float/half conversions

declare <16 x float> @__half_to_float_varying(<16 x i16> %v) nounwind readnone
declare <16 x i16> @__float_to_half_varying(<16 x float> %v) nounwind readnone
declare float @__half_to_float_uniform(i16 %v) nounwind readnone
declare i16 @__float_to_half_uniform(float %v) nounwind readnone

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather

declare void @llvm.trap() noreturn nounwind

gen_gather(i8)
gen_gather(i16)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int32 gathers

declare <16 x i32> @__gather_base_offsets32_i32(i8 * %ptr,
                             i32 %scale, <16 x i32> %offsets,
                             <16 x MASK> %vecmask) nounwind readonly alwaysinline
declare <16 x i32> @__gather_base_offsets64_i32(i8 * %ptr,
                             i32 %scale, <16 x i64> %offsets,
                             <16 x MASK> %vecmask) nounwind readonly alwaysinline
declare <16 x i32> @__gather32_i32(<16 x i32> %ptrs, 
                                   <16 x MASK> %vecmask) nounwind readonly alwaysinline
declare <16 x i32> @__gather64_i32(<16 x i64> %ptrs, 
                                   <16 x MASK> %vecmask) nounwind readonly alwaysinline


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float gathers

declare <16 x float> @__gather_base_offsets32_float(i8 * %ptr,
                                  i32 %scale, <16 x i32> %offsets,
                                  <16 x MASK> %vecmask) nounwind readonly alwaysinline
declare <16 x float> @__gather_base_offsets64_float(i8 * %ptr,
                                   i32 %scale, <16 x i64> %offsets,
                                   <16 x MASK> %vecmask) nounwind readonly alwaysinline
declare <16 x float> @__gather32_float(<16 x i32> %ptrs, 
                                       <16 x MASK> %vecmask) nounwind readonly alwaysinline
declare <16 x float> @__gather64_float(<16 x i64> %ptrs, 
                                       <16 x MASK> %vecmask) nounwind readonly alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int64 gathers

declare <16 x i64> @__gather_base_offsets32_i64(i8 * %ptr,
                             i32 %scale, <16 x i32> %offsets,
                             <16 x MASK> %mask) nounwind readonly alwaysinline
declare <16 x i64> @__gather_base_offsets64_i64(i8 * %ptr,
                             i32 %scale, <16 x i64> %offsets,
                             <16 x MASK> %mask) nounwind readonly alwaysinline

declare <16 x i64> @__gather32_i64(<16 x i32> %ptrs, 
                                   <16 x MASK> %mask) nounwind readonly alwaysinline
declare <16 x i64> @__gather64_i64(<16 x i64> %ptrs, 
                                   <16 x MASK> %mask) nounwind readonly alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double gathers

declare <16 x double> @__gather_base_offsets32_double(i8 * %ptr,
                             i32 %scale, <16 x i32> %offsets,
                             <16 x MASK> %mask32) nounwind readonly alwaysinline
declare <16 x double> @__gather_base_offsets64_double(i8 * %ptr,
                             i32 %scale, <16 x i64> %offsets,
                             <16 x MASK> %mask32) nounwind readonly alwaysinline

declare <16 x double> @__gather32_double(<16 x i32> %ptrs, 
                                         <16 x MASK> %mask32) nounwind readonly alwaysinline
declare <16 x double> @__gather64_double(<16 x i64> %ptrs, 
                                         <16 x MASK> %mask32) nounwind readonly alwaysinline
