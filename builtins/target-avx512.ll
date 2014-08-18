;;  Copyright (c) 2014 Intel Corporation
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

;; TODO:
;; - round/ceil/floor uniform/varying double
;; - reduce_add/min/max_float, reduce_add_int8, reduce_add_double
;; - maked_store_i8/16
;; - Use vector ctlz, cttz for varying versions of these in stdlib--currently,
;;   these dispatch out to do one lane at a time.  There are LLVM intrinsics
;;   for these now, so can we just use those for everything?
;; - proper packed store active... vexpandps, vcompresspd (need LLVM support)
;; - Transcendentals: vexp2
;; - vfixupimmps ?
;; - vpconflict for atomics?
;; - round/ceil/floor uniform float (waiting on http://llvm.org/bugs/show_bug.cgi?id=20684)
;; - 64-bit masked load bug: http://llvm.org/bugs/show_bug.cgi?id=20677
;; - Can't pass <16 x i1> to function bug: http://llvm.org/bugs/show_bug.cgi?id=20665
;; - AVX1/2 and SSE return 12 bits of precision from single precision rcp and rsqrt,
;;   but AVX512 gives 14. Consequently, we don't currently do a Newton-Raphson
;;   step to improve the results.  Should we?
;; - rcp14 vs rcp28, ditto for rsqrt

define(`WIDTH',`16')
define(`MASK',`i1')
define(`HAVE_GATHER',`1')
define(`HAVE_SCATTER',`1')

include(`util.m4')

;; FIXME: could use v16tov 8 (minus space...)
;; similarly many others...
;; $1: mask, $2: low 8 bits, $3: high 8 bits
define(`split_mask', `
  %m16_split = bitcast <16 x MASK> $1 to i16
  $2 = trunc i16 %m16_split to i8
  $3_a = lshr i16 %m16_split, 8
  $3 = trunc i16 $3_a to i8
')

stdlib_core()
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

; operands: source, value if mask off, mask
declare <16 x float> @llvm.x86.avx512.rcp14.ps.512(<16 x float>, <16 x float>,
                                                   i16) nounwind readnone

define <16 x float> @__rcp_varying_float(<16 x float>) nounwind readonly alwaysinline {
  %r = call <16 x float> @llvm.x86.avx512.rcp14.ps.512(<16 x float> %0,
        <16 x float> zeroinitializer, i16 -1)
  ret <16 x float> %r
}

declare <4 x float> @llvm.x86.avx512.rcp14.ss(<4 x float>, <4 x float>, <4 x float>,
                                              i8) nounwind readnone

define float @__rcp_uniform_float(float) nounwind readonly alwaysinline {
  %fv = bitcast float %0 to <1 x float>
  %v = shufflevector <1 x float> %fv, <1 x float> undef,
	  <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
  %rcpv = call <4 x float> @llvm.x86.avx512.rcp14.ss(<4 x float> %v, <4 x float> %v,
        <4 x float> zeroinitializer, i8 -1) 
  %rcp = extractelement <4 x float> %rcpv, i32 0
  ret float %rcp
}

declare <2 x double> @llvm.x86.avx512.rcp14.sd(<2 x double>, <2 x double>, <2 x double>,
                                               i8) nounwind readnone

define double @__rcp_uniform_double(double) nounwind readonly alwaysinline {
  %fv = bitcast double %0 to <1 x double>
  %v = shufflevector <1 x double> %fv, <1 x double> undef,
	  <2 x i32> <i32 0, i32 undef>
  %rcpv = call <2 x double> @llvm.x86.avx512.rcp14.sd(<2 x double> %v, <2 x double> %v,
        <2 x double> zeroinitializer, i8 -1) 
  %rcp = extractelement <2 x double> %rcpv, i32 0

  ; do one N-R iteration to improve precision
  %v_iv = fmul double %0, %rcp
  %two_minus = fsub double 2., %v_iv  
  %iv_mul = fmul double %rcp, %two_minus
  ret double %iv_mul
}

declare <8 x double> @llvm.x86.avx512.rcp14.pd.512(<8 x double>, <8 x double>,
                                                   i8) nounwind readnone

define <16 x double> @__rcp_varying_double(<16 x double>) nounwind readonly alwaysinline {
  v16tov8(double, %0, %v0, %v1)

  %r0 = call <8 x double> @llvm.x86.avx512.rcp14.pd.512(<8 x double> %v0,
               <8 x double> zeroinitializer, i8 -1)
  %r1 = call <8 x double> @llvm.x86.avx512.rcp14.pd.512(<8 x double> %v1,
               <8 x double> zeroinitializer, i8 -1)

  %r = shufflevector <8 x double> %r0, <8 x double> %r1,
           <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                       i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  %v_iv = fmul <16 x double> %0, %r
  %two_minus = fsub <16 x double> <double 2., double 2., double 2., double 2.,
                                   double 2., double 2., double 2., double 2.,
                                   double 2., double 2., double 2., double 2.,
                                   double 2., double 2., double 2., double 2.>, %v_iv  
  %iv_mul = fmul <16 x double> %r, %two_minus
  ret <16 x double> %iv_mul
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding floats

;; args: value to round, (value again), rounding mode
declare <4 x float> @llvm.x86.avx512.rndscale.ss(<4 x float>, <4 x float>, i32)

define float @__round_uniform_float(float) nounwind readonly alwaysinline {
   %fv = bitcast float %0 to <1 x float>
   %v = shufflevector <1 x float> %fv, <1 x float> undef,
	  <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
   %rv = call <4 x float> @llvm.x86.avx512.rndscale.ss(<4 x float> %v,
          <4 x float> %v, i32 0)
   %r = extractelement <4 x float> %rv, i32 0
   ret float %r
}

declare float @__floor_uniform_float(float) nounwind readonly alwaysinline
declare float @__ceil_uniform_float(float) nounwind readonly alwaysinline

;; args: value to round, rounding mode, (value again), mask, XX [4]
declare <16 x float> @llvm.x86.avx512.mask.rndscale.ps.512(<16 x float>, i32,
    <16 x float>, i16, i32)

define <16 x float> @__round_varying_float(<16 x float>) nounwind readonly alwaysinline {
  ; i32 0 -> round to nearest even, no scale, no precision check 
  %r = call <16 x float> @llvm.x86.avx512.mask.rndscale.ps.512(<16 x float> %0, i32 0,
    <16 x float> %0, i16 -1, i32 4)
  ret <16 x float> %r
}

define <16 x float> @__floor_varying_float(<16 x float>) nounwind readonly alwaysinline {
  ; i32 1 -> round to equal or smaller, no scale, no precision check 
  %r = call <16 x float> @llvm.x86.avx512.mask.rndscale.ps.512(<16 x float> %0, i32 1,
    <16 x float> %0, i16 -1, i32 4)
  ret <16 x float> %r
}

define <16 x float> @__ceil_varying_float(<16 x float>) nounwind readonly alwaysinline {
  ; i32 2 -> round to equal or larger, no scale, no precision check 
  %r = call <16 x float> @llvm.x86.avx512.mask.rndscale.ps.512(<16 x float> %0, i32 2,
    <16 x float> %0, i16 -1, i32 4)
  ret <16 x float> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rounding doubles

declare <16 x double> @__round_varying_double(<16 x double>) nounwind readonly alwaysinline
declare <16 x double> @__floor_varying_double(<16 x double>) nounwind readonly alwaysinline
declare <16 x double> @__ceil_varying_double(<16 x double>) nounwind readonly alwaysinline

declare double @__round_uniform_double(double) nounwind readonly alwaysinline
declare double @__floor_uniform_double(double) nounwind readonly alwaysinline
declare double @__ceil_uniform_double(double) nounwind readonly alwaysinline

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; rsqrt

; val, val, zeroinitializer, -1
declare <4 x float> @llvm.x86.avx512.rsqrt14.ss(<4 x float>, <4 x float>, <4 x float>,
                                                i8) nounwind readnone
declare <2 x double> @llvm.x86.avx512.rsqrt14.sd(<2 x double>, <2 x double>, <2 x double>,
                                                 i8) nounwind readnone

define float @__rsqrt_uniform_float(float) nounwind readonly alwaysinline {
  %fv = bitcast float %0 to <1 x float>
  %v = shufflevector <1 x float> %fv, <1 x float> undef,
	  <4 x i32> <i32 0, i32 undef, i32 undef, i32 undef>
  %rv = call <4 x float> @llvm.x86.avx512.rsqrt14.ss(<4 x float> %v, <4 x float> %v,
                <4 x float> zeroinitializer, i8 -1)
  %r = extractelement <4 x float> %rv, i32 0
  ret float %r
}

define double @__rsqrt_uniform_double(double) nounwind readonly alwaysinline {
  %fv = bitcast double %0 to <1 x double>
  %v = shufflevector <1 x double> %fv, <1 x double> undef,
	  <2 x i32> <i32 0, i32 undef>
  %rv = call <2 x double> @llvm.x86.avx512.rsqrt14.sd(<2 x double> %v, <2 x double> %v,
                <2 x double> zeroinitializer, i8 -1)
  %is = extractelement <2 x double> %rv, i32 0

  ; Newton-Raphson iteration to improve precision
  ;  return 0.5 * is * (3. - (v * is) * is);
  %v_is = fmul double %0, %is
  %v_is_is = fmul double %v_is, %is
  %three_sub = fsub double 3., %v_is_is
  %is_mul = fmul double %is, %three_sub
  %half_scale = fmul double 0.5, %is_mul
  ret double %half_scale
}

;; val, zeroinitializer, mask
declare <16 x float> @llvm.x86.avx512.rsqrt14.ps.512(<16 x float>, <16 x float>,
                                                     i16) nounwind readnone
declare <8 x double> @llvm.x86.avx512.rsqrt14.pd.512(<8 x double>, <8 x double>,
                                                     i8) nounwind readnone

define <16 x float> @__rsqrt_varying_float(<16 x float>) nounwind readonly alwaysinline {
  %r = call <16 x float> @llvm.x86.avx512.rsqrt14.ps.512(<16 x float> %0,
                <16 x float> zeroinitializer, i16 -1) nounwind 
  ret <16 x float> %r
}

define <16 x double> @__rsqrt_varying_double(<16 x double>) nounwind readonly alwaysinline {
  v16tov8(double, %0, %v0, %v1)

  %r0 = call <8 x double> @llvm.x86.avx512.rsqrt14.pd.512(<8 x double> %v0,
          <8 x double> zeroinitializer, i8 -1)
  %r1 = call <8 x double> @llvm.x86.avx512.rsqrt14.pd.512(<8 x double> %v1,
          <8 x double> zeroinitializer, i8 -1)

  %is = shufflevector <8 x double> %r0, <8 x double> %r1,
           <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                       i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>


  ;  return 0.5 * is * (3. - (v * is) * is);
  %v_is = fmul <16 x double> %0, %is
  %v_is_is = fmul <16 x double> %v_is, %is
  %three_sub = fsub <16 x double> <double 3., double 3., double 3., double 3.,
                                   double 3., double 3., double 3., double 3.,
                                   double 3., double 3., double 3., double 3.,
                                   double 3., double 3., double 3., double 3.>, %v_is_is
  %is_mul = fmul <16 x double> %is, %three_sub
  %half_scale = fmul <16 x double> <double 0.5, double 0.5, double 0.5, double 0.5,
                                    double 0.5, double 0.5, double 0.5, double 0.5,
                                    double 0.5, double 0.5, double 0.5, double 0.5,
                                    double 0.5, double 0.5, double 0.5, double 0.5>, %is_mul
  ret <16 x double> %half_scale
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; sqrt

declare float @sqrtf(float) readnone

define float @__sqrt_uniform_float(float) nounwind readonly alwaysinline {
  %v = call float @sqrtf(float %0)
  ret float %v
}

declare <16 x float> @llvm.sqrt.v16f32(<16 x float>)

define <16 x float> @__sqrt_varying_float(<16 x float>) nounwind readonly alwaysinline {
  %r = call <16 x float> @llvm.sqrt.v16f32(<16 x float> %0)
  ret <16 x float> %r
}

declare double @sqrt(double) readnone

define double @__sqrt_uniform_double(double) nounwind alwaysinline {
  %v = call double @sqrt(double %0)
  ret double %v
}

declare <8 x double> @llvm.sqrt.v8f64(<8 x double>)

define <16 x double> @__sqrt_varying_double(<16 x double>) nounwind alwaysinline {
  unary8to16(r, double, @llvm.sqrt.v8f64, %0)
  ret <16 x double> %r
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

define float @__max_uniform_float(float, float) nounwind readonly alwaysinline {
  %cmp = fcmp ogt float %1, %0
  %ret = select i1 %cmp, float %1, float %0
  ret float %ret
}

define float @__min_uniform_float(float, float) nounwind readonly alwaysinline {
  %cmp = fcmp ogt float %1, %0
  %ret = select i1 %cmp, float %0, float %1
  ret float %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision min/max

define double @__min_uniform_double(double, double) nounwind readnone alwaysinline {
  %cmp = fcmp ogt double %1, %0
  %ret = select i1 %cmp, double %0, double %1
  ret double %ret
}

define double @__max_uniform_double(double, double) nounwind readnone alwaysinline {
  %cmp = fcmp ogt double %1, %0
  %ret = select i1 %cmp, double %1, double %0
  ret double %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int min/max

define i32 @__min_uniform_int32(i32, i32) nounwind readonly alwaysinline {
  %cmp = icmp sgt i32 %1, %0
  %ret = select i1 %cmp, i32 %0, i32 %1
  ret i32 %ret
}

define i32 @__max_uniform_int32(i32, i32) nounwind readonly alwaysinline {
  %cmp = icmp sgt i32 %1, %0
  %ret = select i1 %cmp, i32 %1, i32 %0
  ret i32 %ret
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; unsigned int min/max

define i32 @__min_uniform_uint32(i32, i32) nounwind readonly alwaysinline {
  %cmp = icmp ugt i32 %1, %0
  %ret = select i1 %cmp, i32 %0, i32 %1
  ret i32 %ret
}

define i32 @__max_uniform_uint32(i32, i32) nounwind readonly alwaysinline {
  %cmp = icmp ugt i32 %1, %0
  %ret = select i1 %cmp, i32 %1, i32 %0
  ret i32 %ret
}

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
;; int8/int16 builtins

define_avgs()

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; min/max

;; arguments: first val, second val, val for mask=0 lanes, mask, ???
;; TODO: I believe that the last operand is related to the rounding 
;; variants of this instruction, which are spectacularly undocumented
;; in the AVX512 spec--they are mentioned, but never explained.
declare <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float>, <16 x float>,
                    <16 x float>, i16, i32)
declare <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float>, <16 x float>,
                    <16 x float>, i16, i32)

define <16 x float> @__max_varying_float(<16 x float>,
                                         <16 x float>) nounwind readonly alwaysinline {
  %r = call <16 x float> @llvm.x86.avx512.mask.max.ps.512(<16 x float> %0, <16 x float> %1,
                    <16 x float> zeroinitializer, i16 -1, i32 4)
  ret <16 x float> %r
}

define <16 x float> @__min_varying_float(<16 x float>,
                                         <16 x float>) nounwind readonly alwaysinline {
  %r = call <16 x float> @llvm.x86.avx512.mask.min.ps.512(<16 x float> %0, <16 x float> %1,
                    <16 x float> zeroinitializer, i16 -1, i32 4)
  ret <16 x float> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; horizontal ops

define i64 @__movmsk(<16 x MASK>) nounwind readnone alwaysinline {
  %m16 = bitcast <16 x MASK> %0 to i16
  %m = zext i16 %m16 to i64
  ret i64 %m
}

define i1 @__any(<16 x MASK>) nounwind readnone alwaysinline {
  %m16 = bitcast <16 x MASK> %0 to i16
  %c = icmp ne i16 %m16, 0
  ret i1 %c
}

define i1 @__all(<16 x MASK>) nounwind readnone alwaysinline {
  %m16 = bitcast <16 x MASK> %0 to i16
  %c = icmp eq i16 %m16, 65535
  ret i1 %c
}

define i1 @__none(<16 x MASK>) nounwind readnone alwaysinline {
  %m16 = bitcast <16 x MASK> %0 to i16
  %c = icmp eq i16 %m16, 0
  ret i1 %c
}

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

declare <16 x float> @llvm.x86.avx512.mask.loadu.ps.512(i8*, <16 x float>, i16 )
declare <8 x double> @llvm.x86.avx512.mask.loadu.pd.512(i8*, <8 x double>, i8 )

define <16 x float> @__masked_load_float(i8 *, <16 x MASK> %mask) nounwind alwaysinline {
  %m16 = bitcast <16 x MASK> %mask to i16
  %r = call <16 x float> @llvm.x86.avx512.mask.loadu.ps.512(i8 * %0, <16 x float> zeroinitializer, i16 %m16)
  ret <16 x float> %r
}

define <16 x double> @__masked_load_double(i8 *, <16 x MASK> %mask) nounwind alwaysinline {
  v16tov8(i1, %mask, %maska, %maskb)
  %maska_i8 = bitcast <8 x i1> %maska to i8
  %maskb_i8 = bitcast <8 x i1> %maskb to i8
  %ra = call <8 x double> @llvm.x86.avx512.mask.loadu.pd.512(i8* %0, <8 x double> zeroinitializer, i8 %maska_i8)
  %ptrb = getelementptr i8 * %0, i64 512
  %rb = call <8 x double> @llvm.x86.avx512.mask.loadu.pd.512(i8* %ptrb, <8 x double> zeroinitializer, i8 %maskb_i8)
  %r16 = shufflevector <8 x double> %ra, <8 x double> %rb,
           <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
                       i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x double> %r16
}

define <16 x i32> @__masked_load_i32(i8 *, <16 x MASK> %mask) nounwind alwaysinline {
  %v = call <16 x float> @__masked_load_float(i8 * %0, <16 x MASK> %mask)
  %vi = bitcast <16 x float> %v to <16 x i32>
  ret <16 x i32> %vi
}

define <16 x i64> @__masked_load_i64(i8 *, <16 x MASK> %mask) nounwind alwaysinline {
  %v = call <16 x double> @__masked_load_double(i8 * %0, <16 x MASK> %mask)
  %vi = bitcast <16 x double> %v to <16 x i64>
  ret <16 x i64> %vi
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; masked store

declare void @llvm.x86.avx512.mask.storeu.ps.512(i8*, <16 x float>, i16 )
declare void @llvm.x86.avx512.mask.storeu.pd.512(i8*, <8 x double>, i8 )

declare void @__masked_store_i8(<16 x i8>* nocapture %ptr, <16 x i8> %val, 
                                <16 x MASK> %mask) nounwind alwaysinline
declare void @__masked_store_i16(<16 x i16>* nocapture %ptr, <16 x i16> %val,
                                 <16 x MASK> %mask) nounwind alwaysinline

define void @__masked_store_i32(<16 x i32>* nocapture %ptr, <16 x i32> %val, 
                                <16 x MASK> %mask) nounwind alwaysinline {
  %vf = bitcast <16 x i32> %val to <16 x float>
  %m16 = bitcast <16 x MASK> %mask to i16
  %ptr8 = bitcast <16 x i32> * %ptr to i8 *
  call void @llvm.x86.avx512.mask.storeu.ps.512(i8* %ptr8, <16 x float> %vf, i16 %m16)
  ret void
}

define void @__masked_store_i64(<16 x i64>* nocapture %ptr, <16 x i64> %val,
                                <16 x MASK> %mask) nounwind alwaysinline {
  split_mask(%mask, %m0, %m1)
  
  %p0 = bitcast <16 x i64> * %ptr to i8 *
  %p1 = getelementptr i8 * %p0, i64 512

  %v0 = shufflevector <16 x i64> %val, <16 x i64> undef,
    <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v1 = shufflevector <16 x i64> %val, <16 x i64> undef,
    <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  %v0d = bitcast <8 x i64> %v0 to <8 x double>
  %v1d = bitcast <8 x i64> %v1 to <8 x double>

  call void @llvm.x86.avx512.mask.storeu.pd.512(i8* %p0, <8 x double> %v0d, i8 %m0)
  call void @llvm.x86.avx512.mask.storeu.pd.512(i8* %p1, <8 x double> %v1d, i8 %m1)
  ret void
}

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


gen_scatter(i8)
gen_scatter(i16)

define void @__scatter_base_offsets32_i8(i8* %ptr, i32 %scale, <WIDTH x i32> %offsets,
                                         <WIDTH x i8> %vals, <WIDTH x i1> %mask) nounwind {
  call void @__scatter_factored_base_offsets32_i8(i8* %ptr, <16 x i32> %offsets,
      i32 %scale, <16 x i32> zeroinitializer, <16 x i8> %vals, <WIDTH x i1> %mask)
  ret void
}

define void @__scatter_base_offsets64_i8(i8* %ptr, i32 %scale, <WIDTH x i64> %offsets,
                                         <WIDTH x i8> %vals, <WIDTH x i1> %mask) nounwind {
  call void @__scatter_factored_base_offsets64_i8(i8* %ptr, <16 x i64> %offsets,
      i32 %scale, <16 x i64> zeroinitializer, <16 x i8> %vals, <WIDTH x i1> %mask)
  ret void
}

define void @__scatter_base_offsets32_i16(i8* %ptr, i32 %scale, <WIDTH x i32> %offsets,
                                          <WIDTH x i16> %vals, <WIDTH x i1> %mask) nounwind {
  call void @__scatter_factored_base_offsets32_i16(i8* %ptr, <16 x i32> %offsets,
      i32 %scale, <16 x i32> zeroinitializer, <16 x i16> %vals, <WIDTH x i1> %mask)
  ret void
}

define void @__scatter_base_offsets64_i16(i8* %ptr, i32 %scale, <WIDTH x i64> %offsets,
                                          <WIDTH x i16> %vals, <WIDTH x i1> %mask) nounwind {
  call void @__scatter_factored_base_offsets64_i16(i8* %ptr, <16 x i64> %offsets,
      i32 %scale, <16 x i64> zeroinitializer, <16 x i16> %vals, <WIDTH x i1> %mask)
  ret void
}

declare void @llvm.x86.avx512.scatter.dps.512(i8*, i16, <16 x i32>, <16 x float>, i32)
declare void @llvm.x86.avx512.scatter.qps.512(i8*, i8, <8 x i64>, <8 x float>, i32)

define void @__scatter_base_offsets32_float(i8* %ptr, i32 %scale, <WIDTH x i32> %offsets,
                                          <WIDTH x float> %val, <WIDTH x i1> %mask) nounwind {
  %m16 = bitcast <WIDTH x i1> %mask to i16
  call void @llvm.x86.avx512.scatter.dps.512 (i8* %ptr, i16 %m16, <16 x i32> %offsets,
      <16 x float> %val, i32 %scale)
  ret void
}

define void @__scatter_base_offsets64_float(i8* %ptr, i32 %scale, <WIDTH x i64> %offsets,
                                          <WIDTH x float> %vals, <WIDTH x i1> %mask) nounwind {
  split_mask(%mask, %m0, %m1)

  %o0 = shufflevector <16 x i64> %offsets, <16 x i64> undef,
    <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %o1 = shufflevector <16 x i64> %offsets, <16 x i64> undef,
    <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  %v0 = shufflevector <16 x float> %vals, <16 x float> undef,
    <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v1 = shufflevector <16 x float> %vals, <16 x float> undef,
    <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  call void @llvm.x86.avx512.scatter.qps.512(i8* %ptr, i8 %m0, <8 x i64> %o0,
               <8 x float> %v0, i32 %scale)
  call void @llvm.x86.avx512.scatter.qps.512(i8* %ptr, i8 %m1, <8 x i64> %o1,
               <8 x float> %v1, i32 %scale)
  ret void
}

define void @__scatter32_float(<WIDTH x i32> %ptrs, <WIDTH x float> %vals,
                               <WIDTH x i1> %mask) nounwind {
  call void @__scatter_base_offsets32_float(i8 * zeroinitializer, i32 1,
             <16 x i32> %ptrs, <16 x float> %vals, <16 x i1> %mask)
  ret void
}

define void @__scatter64_float(<WIDTH x i64> %ptrs, <WIDTH x float> %vals,
                               <WIDTH x i1> %mask) nounwind {
  call void @__scatter_base_offsets64_float(i8 * zeroinitializer, i32 1,
             <16 x i64> %ptrs, <16 x float> %vals, <16 x i1> %mask)
  ret void
}

define void @__scatter_base_offsets32_i32(i8* nocapture %ptr, i32 %scale,
                  <WIDTH x i32> %offsets, <WIDTH x i32> %vals, <WIDTH x i1> %mask) nounwind  {
  %fvals = bitcast <16 x i32> %vals to <16 x float>
  call void @__scatter_base_offsets32_float(i8* nocapture %ptr, i32 %scale,
                  <WIDTH x i32> %offsets, <WIDTH x float> %fvals, <WIDTH x i1> %mask)
  ret void
}

define void @__scatter_base_offsets64_i32(i8* nocapture %ptr, i32 %scale,
                  <WIDTH x i64> %offsets, <WIDTH x i32> %vals, <WIDTH x i1> %mask) nounwind {
  %fvals = bitcast <16 x i32> %vals to <16 x float>
  call void @__scatter_base_offsets64_float(i8* nocapture %ptr, i32 %scale,
                  <WIDTH x i64> %offsets, <WIDTH x float> %fvals, <WIDTH x i1> %mask)
  ret void
}

define void @__scatter32_i32(<WIDTH x i32> %ptrs, <WIDTH x i32> %vals,
                             <WIDTH x i1> %mask) nounwind {
  %fvals = bitcast <16 x i32> %vals to <16 x float>
  call void @__scatter32_float(<WIDTH x i32> %ptrs, <WIDTH x float> %fvals,
                               <WIDTH x i1> %mask)
  ret void
}

define void @__scatter64_i32(<WIDTH x i64> %ptrs, <WIDTH x i32> %vals,
                             <WIDTH x i1> %mask) nounwind {
  %fvals = bitcast <16 x i32> %vals to <16 x float>
  call void @__scatter64_float(<WIDTH x i64> %ptrs, <WIDTH x float> %fvals,
                              <WIDTH x i1> %mask)
  ret void
}

declare void @llvm.x86.avx512.scatter.dpd.512(i8*, i8, <8 x i32>, <8 x double>, i32)
declare void @llvm.x86.avx512.scatter.qpd.512(i8*, i8, <8 x i64>, <8 x double>, i32)

define void @__scatter_base_offsets32_double(i8* %ptr, i32 %scale, <WIDTH x i32> %offsets,
                                          <WIDTH x double> %vals, <WIDTH x i1> %mask) nounwind {
  split_mask(%mask, %m0, %m1)

  %o0 = shufflevector <16 x i32> %offsets, <16 x i32> undef,
    <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %o1 = shufflevector <16 x i32> %offsets, <16 x i32> undef,
    <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  %v0 = shufflevector <16 x double> %vals, <16 x double> undef,
    <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v1 = shufflevector <16 x double> %vals, <16 x double> undef,
    <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  call void @llvm.x86.avx512.scatter.dpd.512(i8* %ptr, i8 %m0, <8 x i32> %o0,
               <8 x double> %v0, i32 %scale)
  call void @llvm.x86.avx512.scatter.dpd.512(i8* %ptr, i8 %m1, <8 x i32> %o1,
               <8 x double> %v1, i32 %scale)
  ret void
}

define void @__scatter_base_offsets64_double(i8* %ptr, i32 %scale, <WIDTH x i64> %offsets,
                                         <WIDTH x double> %vals, <WIDTH x i1> %mask) nounwind {
  split_mask(%mask, %m0, %m1)

  %o0 = shufflevector <16 x i64> %offsets, <16 x i64> undef,
    <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %o1 = shufflevector <16 x i64> %offsets, <16 x i64> undef,
    <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  %v0 = shufflevector <16 x double> %vals, <16 x double> undef,
    <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v1 = shufflevector <16 x double> %vals, <16 x double> undef,
    <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  call void @llvm.x86.avx512.scatter.qpd.512(i8* %ptr, i8 %m0, <8 x i64> %o0,
               <8 x double> %v0, i32 %scale)
  call void @llvm.x86.avx512.scatter.qpd.512(i8* %ptr, i8 %m1, <8 x i64> %o1,
               <8 x double> %v1, i32 %scale)
  ret void
}

define void @__scatter32_double(<WIDTH x i32> %ptrs, <WIDTH x double> %vals,
                                <WIDTH x i1> %mask) nounwind {
  call void @__scatter_base_offsets32_double(i8 * zeroinitializer, i32 1,
             <16 x i32> %ptrs, <16 x double> %vals, <16 x i1> %mask)
  ret void
}

define void @__scatter64_double(<WIDTH x i64> %ptrs, <WIDTH x double> %vals,
                                <WIDTH x i1> %mask) nounwind {
  call void @__scatter_base_offsets64_double(i8 * zeroinitializer, i32 1,
             <16 x i64> %ptrs, <16 x double> %vals, <16 x i1> %mask)
  ret void
}

define void @__scatter_base_offsets32_i64(i8* nocapture %ptr, i32 %scale,
                  <WIDTH x i32> %offsets, <WIDTH x i64> %vals, <WIDTH x i1> %mask) nounwind  {
  %fvals = bitcast <16 x i64> %vals to <16 x double>
  call void @__scatter_base_offsets32_double(i8* nocapture %ptr, i32 %scale,
                  <WIDTH x i32> %offsets, <WIDTH x double> %fvals, <WIDTH x i1> %mask)
  ret void
}

define void @__scatter_base_offsets64_i64(i8* nocapture %ptr, i32 %scale,
                  <WIDTH x i64> %offsets, <WIDTH x i64> %vals, <WIDTH x i1> %mask) nounwind {
  %fvals = bitcast <16 x i64> %vals to <16 x double>
  call void @__scatter_base_offsets64_double(i8* nocapture %ptr, i32 %scale,
                  <WIDTH x i64> %offsets, <WIDTH x double> %fvals, <WIDTH x i1> %mask)
  ret void
}

define void @__scatter32_i64(<WIDTH x i32> %ptrs, <WIDTH x i64> %vals,
                             <WIDTH x i1> %mask) nounwind {
  %fvals = bitcast <16 x i64> %vals to <16 x double>
  call void @__scatter32_double(<WIDTH x i32> %ptrs, <WIDTH x double> %fvals,
                               <WIDTH x i1> %mask)
  ret void
}

define void @__scatter64_i64(<WIDTH x i64> %ptrs, <WIDTH x i64> %vals,
                             <WIDTH x i1> %mask) nounwind {
  %fvals = bitcast <16 x i64> %vals to <16 x double>
  call void @__scatter64_double(<WIDTH x i64> %ptrs, <WIDTH x double> %fvals,
                                <WIDTH x i1> %mask)
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double precision min/max

declare <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double>, <8 x double>,
                    <8 x double>, i8, i32)
declare <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double>, <8 x double>,
                    <8 x double>, i8, i32)

define <16 x double> @__min_varying_double(<16 x double> %a,
           <16 x double> %b) nounwind readnone alwaysinline {
  %a0 = shufflevector <16 x double> %a, <16 x double> undef,
    <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %a1 = shufflevector <16 x double> %a, <16 x double> undef,
    <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %b0 = shufflevector <16 x double> %b, <16 x double> undef,
    <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %b1 = shufflevector <16 x double> %b, <16 x double> undef,
    <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  %r0 = call <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double> %a0, <8 x double> %b0,
                    <8 x double> zeroinitializer, i8 -1, i32 4)
  %r1 = call <8 x double> @llvm.x86.avx512.mask.min.pd.512(<8 x double> %a1, <8 x double> %b1,
                    <8 x double> zeroinitializer, i8 -1, i32 4)

  %r = shufflevector <8 x double> %r0, <8 x double> %r1,
      <16 x i32> <i32  0, i32 1,  i32  2, i32  3, 
                  i32  4, i32 5,  i32  6, i32  7,
                  i32  8, i32 9,  i32 10, i32 11,
                  i32 12, i32 13, i32 14, i32 15>
  ret <16 x double> %r
}

define <16 x double> @__max_varying_double(<16 x double> %a,
           <16 x double> %b) nounwind readnone alwaysinline {
  %a0 = shufflevector <16 x double> %a, <16 x double> undef,
    <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %a1 = shufflevector <16 x double> %a, <16 x double> undef,
    <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %b0 = shufflevector <16 x double> %b, <16 x double> undef,
    <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %b1 = shufflevector <16 x double> %b, <16 x double> undef,
    <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>

  %r0 = call <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double> %a0, <8 x double> %b0,
                    <8 x double> zeroinitializer, i8 -1, i32 4)
  %r1 = call <8 x double> @llvm.x86.avx512.mask.max.pd.512(<8 x double> %a1, <8 x double> %b1,
                    <8 x double> zeroinitializer, i8 -1, i32 4)

  %r = shufflevector <8 x double> %r0, <8 x double> %r1,
      <16 x i32> <i32  0, i32 1,  i32  2, i32  3, 
                  i32  4, i32 5,  i32  6, i32  7,
                  i32  8, i32 9,  i32 10, i32 11,
                  i32 12, i32 13, i32 14, i32 15>
  ret <16 x double> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int/unsigned int min/max

declare <16 x i32> @llvm.x86.avx512.mask.pmins.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)
declare <16 x i32> @llvm.x86.avx512.mask.pmaxs.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

define <16 x i32> @__min_varying_int32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  %r = call <16 x i32> @llvm.x86.avx512.mask.pmins.d.512(<16 x i32> %0, <16 x i32> %1,
            <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %r
}

define <16 x i32> @__max_varying_int32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  %r = call <16 x i32> @llvm.x86.avx512.mask.pmaxs.d.512(<16 x i32> %0, <16 x i32> %1,
            <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %r
}

declare <16 x i32> @llvm.x86.avx512.mask.pminu.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)
declare <16 x i32> @llvm.x86.avx512.mask.pmaxu.d.512(<16 x i32>, <16 x i32>, <16 x i32>, i16)

define <16 x i32> @__min_varying_uint32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  %r = call <16 x i32> @llvm.x86.avx512.mask.pminu.d.512(<16 x i32> %0, <16 x i32> %1,
            <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %r
}

define <16 x i32> @__max_varying_uint32(<16 x i32>, <16 x i32>) nounwind readonly alwaysinline {
  %r = call <16 x i32> @llvm.x86.avx512.mask.pmaxu.d.512(<16 x i32> %0, <16 x i32> %1,
            <16 x i32> zeroinitializer, i16 -1)
  ret <16 x i32> %r
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float/half conversions

declare <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16>, <16 x float>,
    i16, i32) nounwind readonly
declare <16 x i16> @llvm.x86.avx512.mask.vcvtps2ph.512(<16 x float>, i32, <16 x i16>,
    i16) nounwind readonly

define <16 x float> @__half_to_float_varying(<16 x i16> %v) nounwind readnone {
  %r = call <16 x float> @llvm.x86.avx512.mask.vcvtph2ps.512(<16 x i16> %v,
              <16 x float> zeroinitializer, i16 -1, i32 4) nounwind readonly
  ret <16 x float> %r
}

define <16 x i16> @__float_to_half_varying(<16 x float> %v) nounwind readnone {
  ;; i32 0 -> round to nearest even
  %r = call <16 x i16> @llvm.x86.avx512.mask.vcvtps2ph.512(<16 x float> %v,
               i32 0, <16 x i16> zeroinitializer, i16 -1)
  ret <16 x i16> %r
}

define float @__half_to_float_uniform(i16 %v) nounwind readnone {
  %v1 = bitcast i16 %v to <1 x i16>
  %vv = shufflevector <1 x i16> %v1, <1 x i16> undef,
        <16 x i32> <i32 0, i32 undef, i32 undef, i32 undef,
                    i32 undef, i32 undef, i32 undef, i32 undef, 
                    i32 undef, i32 undef, i32 undef, i32 undef, 
                    i32 undef, i32 undef, i32 undef, i32 undef>
  %vh = call <16 x float> @__half_to_float_varying(<16 x i16> %vv)
  %h = extractelement <16 x float> %vh, i32 0
  ret float %h
}

define i16 @__float_to_half_uniform(float %v) nounwind readnone {
  %v1 = bitcast float %v to <1 x float>
  %vv = shufflevector <1 x float> %v1, <1 x float> undef,
        <16 x i32> <i32 0, i32 undef, i32 undef, i32 undef,
                    i32 undef, i32 undef, i32 undef, i32 undef, 
                    i32 undef, i32 undef, i32 undef, i32 undef, 
                    i32 undef, i32 undef, i32 undef, i32 undef>
  %vh = call <16 x i16> @__float_to_half_varying(<16 x float> %vv)
  %h = extractelement <16 x i16> %vh, i32 0
  ret i16 %h
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; gather

declare void @llvm.trap() noreturn nounwind

gen_gather(i8)
gen_gather(i16)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; float gathers

;; src, base, indices, mask, scale
declare <16 x float> @llvm.x86.avx512.gather.dps.512 (<16 x float>, i8*, <16 x i32>, i16, i32)

define <16 x float> @__gather_base_offsets32_float(i8 * %ptr,
                                  i32 %scale, <16 x i32> %offsets,
                                  <16 x MASK> %mask) nounwind readonly alwaysinline {
  %m16 = bitcast <16 x MASK> %mask to i16
  %v = call <16 x float> @llvm.x86.avx512.gather.dps.512(<16 x float> undef, i8* %ptr,
  	<16 x i32> %offsets, i16 %m16, i32 %scale)
  ret <16 x float> %v
}

declare <8 x float> @llvm.x86.avx512.gather.qps.512(<8 x float>, i8*, <8 x i64>, i8, i32)

define <16 x float> @__gather_base_offsets64_float(i8 * %ptr,
                                   i32 %scale, <16 x i64> %offsets,
                                   <16 x MASK> %mask) nounwind readonly alwaysinline {
  split_mask(%mask, %m0, %m1)

  %o0 = shufflevector <16 x i64> %offsets, <16 x i64> undef,
    <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %o1 = shufflevector <16 x i64> %offsets, <16 x i64> undef,
    <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  
  %v0 = call <8 x float> @llvm.x86.avx512.gather.qps.512(<8 x float> undef, i8* %ptr,
      <8 x i64> %o0, i8 %m0, i32 %scale)
  %v1 = call <8 x float> @llvm.x86.avx512.gather.qps.512(<8 x float> undef, i8* %ptr,
      <8 x i64> %o1, i8 %m1, i32 %scale)

  %v = shufflevector <8 x float> %v0, <8 x float> %v1,
      <16 x i32> <i32  0, i32 1,  i32  2, i32  3, 
                  i32  4, i32 5,  i32  6, i32  7,
                  i32  8, i32 9,  i32 10, i32 11,
                  i32 12, i32 13, i32 14, i32 15>
  ret <16 x float> %v
}

define <16 x float> @__gather32_float(<16 x i32> %ptrs, 
                                      <16 x MASK> %mask) nounwind readonly alwaysinline {
  %m16 = bitcast <16 x MASK> %mask to i16
  %v = call <16 x float> @llvm.x86.avx512.gather.dps.512(<16 x float> undef, i8* zeroinitializer,
  	<16 x i32> %ptrs, i16 %m16, i32 1)
  ret <16 x float> %v
}

define <16 x float> @__gather64_float(<16 x i64> %ptrs, 
                                       <16 x MASK> %mask) nounwind readonly alwaysinline {
  %v = call <16 x float> @__gather_base_offsets64_float(i8 * zeroinitializer, i32 1,
                  <16 x i64> %ptrs, <16 x MASK> %mask)
  ret <16 x float> %v
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int32 gathers

define <16 x i32> @__gather_base_offsets32_i32(i8 * %ptr,
                             i32 %scale, <16 x i32> %offsets,
                             <16 x MASK> %mask) nounwind readonly alwaysinline {
  %vf = call <16 x float> @__gather_base_offsets32_float(i8 * %ptr,
                  i32 %scale, <16 x i32> %offsets, <16 x MASK> %mask)
  %v = bitcast <16 x float> %vf to <16 x i32>
  ret <16 x i32> %v
}

define <16 x i32> @__gather_base_offsets64_i32(i8 * %ptr,
                             i32 %scale, <16 x i64> %offsets,
                             <16 x MASK> %mask) nounwind readonly alwaysinline {
  %vf = call <16 x float> @__gather_base_offsets64_float(i8 * %ptr,
                  i32 %scale, <16 x i64> %offsets, <16 x MASK> %mask)
  %v = bitcast <16 x float> %vf to <16 x i32>
  ret <16 x i32> %v
}

define <16 x i32> @__gather32_i32(<16 x i32> %ptrs, 
                                   <16 x MASK> %mask) nounwind readonly alwaysinline {
  %vf = call <16 x float> @__gather32_float(<16 x i32> %ptrs, <16 x MASK> %mask)
  %v = bitcast <16 x float> %vf to <16 x i32>
  ret <16 x i32> %v
}

define <16 x i32> @__gather64_i32(<16 x i64> %ptrs, 
                                   <16 x MASK> %mask) nounwind readonly alwaysinline {
  %vf = call <16 x float> @__gather64_float(<16 x i64> %ptrs, <16 x MASK> %mask)
  %v = bitcast <16 x float> %vf to <16 x i32>
  ret <16 x i32> %v
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; double gathers

declare <8 x double> @llvm.x86.avx512.gather.dpd.512 (<8 x double>, i8*, <8 x i32>, i8, i32)
declare <8 x double> @llvm.x86.avx512.gather.qpd.512 (<8 x double>, i8*, <8 x i64>, i8, i32)

define <16 x double> @__gather_base_offsets32_double(i8 * %ptr,
                             i32 %scale, <16 x i32> %offsets,
                             <16 x MASK> %mask) nounwind readonly alwaysinline {
  split_mask(%mask, %m0, %m1)

  %o0 = shufflevector <16 x i32> %offsets, <16 x i32> undef,
    <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %o1 = shufflevector <16 x i32> %offsets, <16 x i32> undef,
    <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  
  %v0 = call <8 x double> @llvm.x86.avx512.gather.dpd.512(<8 x double> undef,
             i8* %ptr, <8 x i32> %o0, i8 %m0, i32 %scale)
  %v1 = call <8 x double> @llvm.x86.avx512.gather.dpd.512(<8 x double> undef,
             i8* %ptr, <8 x i32> %o1, i8 %m1, i32 %scale)

  %v = shufflevector <8 x double> %v0, <8 x double> %v1,
      <16 x i32> <i32  0, i32 1,  i32  2, i32  3, 
                  i32  4, i32 5,  i32  6, i32  7,
                  i32  8, i32 9,  i32 10, i32 11,
                  i32 12, i32 13, i32 14, i32 15>
  ret <16 x double> %v
}

define <16 x double> @__gather_base_offsets64_double(i8 * %ptr,
                             i32 %scale, <16 x i64> %offsets,
                             <16 x MASK> %mask) nounwind readonly alwaysinline {
  split_mask(%mask, %m0, %m1)

  %o0 = shufflevector <16 x i64> %offsets, <16 x i64> undef,
    <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %o1 = shufflevector <16 x i64> %offsets, <16 x i64> undef,
    <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  
  %v0 = call <8 x double> @llvm.x86.avx512.gather.qpd.512(<8 x double> undef,
             i8* %ptr, <8 x i64> %o0, i8 %m0, i32 %scale)
  %v1 = call <8 x double> @llvm.x86.avx512.gather.qpd.512(<8 x double> undef,
             i8* %ptr, <8 x i64> %o1, i8 %m1, i32 %scale)

  %v = shufflevector <8 x double> %v0, <8 x double> %v1,
      <16 x i32> <i32  0, i32 1,  i32  2, i32  3, 
                  i32  4, i32 5,  i32  6, i32  7,
                  i32  8, i32 9,  i32 10, i32 11,
                  i32 12, i32 13, i32 14, i32 15>
  ret <16 x double> %v
}

define <16 x double> @__gather32_double(<16 x i32> %ptrs, 
                                        <16 x MASK> %mask) nounwind readonly alwaysinline {
  %v = call <16 x double> @__gather_base_offsets32_double(i8 * zeroinitializer,
                             i32 1, <16 x i32> %ptrs, <16 x MASK> %mask)
  ret <16 x double> %v
}

define <16 x double> @__gather64_double(<16 x i64> %ptrs, 
                                         <16 x MASK> %mask) nounwind readonly alwaysinline {
  %v = call <16 x double> @__gather_base_offsets64_double(i8 * zeroinitializer,
                             i32 1, <16 x i64> %ptrs, <16 x MASK> %mask)
  ret <16 x double> %v
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; int64 gathers

define <16 x i64> @__gather_base_offsets32_i64(i8 * %ptr,
                             i32 %scale, <16 x i32> %offsets,
                             <16 x MASK> %mask) nounwind readonly alwaysinline {
  %v = call <16 x double> @__gather_base_offsets32_double(i8 * %ptr,
           i32 %scale, <16 x i32> %offsets, <16 x MASK> %mask)
  %vi = bitcast <16 x double> %v to <16 x i64>
  ret <16 x i64> %vi
}

define <16 x i64> @__gather_base_offsets64_i64(i8 * %ptr,
                             i32 %scale, <16 x i64> %offsets,
                             <16 x MASK> %mask) nounwind readonly alwaysinline {
  %v = call <16 x double> @__gather_base_offsets64_double(i8 * %ptr,
           i32 %scale, <16 x i64> %offsets, <16 x MASK> %mask)
  %vi = bitcast <16 x double> %v to <16 x i64>
  ret <16 x i64> %vi
}

define <16 x i64> @__gather32_i64(<16 x i32> %ptrs, 
                                  <16 x MASK> %mask) nounwind readonly alwaysinline {
  %v = call <16 x double> @__gather32_double(<16 x i32> %ptrs, <16 x MASK> %mask)
  %vi = bitcast <16 x double> %v to <16 x i64>
  ret <16 x i64> %vi
}

define <16 x i64> @__gather64_i64(<16 x i64> %ptrs, 
                                  <16 x MASK> %mask) nounwind readonly alwaysinline {
  %v = call <16 x double> @__gather64_double(<16 x i64> %ptrs, <16 x MASK> %mask)
  %vi = bitcast <16 x double> %v to <16 x i64>
  ret <16 x i64> %vi
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; packed load/store

packed_load_and_store()

;define i32 @__packed_load_active(i32 * %startptr, <WIDTH x i32> * %val_ptr,
;                                 <WIDTH x MASK> %full_mask) nounwind alwaysinline {
;  ret i32 undef
;}
;
;define i32 @__packed_store_active(i32 * %startptr, <WIDTH x i32> %vals,
;                                   <WIDTH x MASK> %full_mask) nounwind alwaysinline {
;  ret i32 undef
;}
;
;define MASK @__packed_store_active2(i32 * %startptr, <WIDTH x i32> %vals,
;                                    <WIDTH x MASK> %full_mask) nounwind alwaysinline {
;  ret MASK undef
;}

