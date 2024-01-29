#map = affine_map<()[s0] -> ((s0 floordiv 4) * 4)>
#map1 = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0)  -> (d0 + 1)>
#map2 = affine_map<(d0) -> (d0 + 2)>
#map3 = affine_map<(d0) -> (d0 + 3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_gemm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f64, %arg4: f64, %arg5: memref<?x1100xf64>, %arg6: memref<?x1200xf64>, %arg7: memref<?x1100xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg2 : i32 to index
    %2 = arith.index_cast %arg0 : i32 to index
    affine.for %arg8 = 0 to %2 {
      affine.for %arg9 = 0 to #map()[%0] step 4 {
        %3 = affine.load %arg5[%arg8, %arg9] : memref<?x1100xf64>
        %4 = arith.mulf %3, %arg4 : f64
        affine.store %4, %arg5[%arg8, %arg9] : memref<?x1100xf64>
        %5 = affine.apply #map1(%arg9)
        %6 = affine.load %arg5[%arg8, %5] : memref<?x1100xf64>
        %7 = arith.mulf %6, %arg4 : f64
        affine.store %7, %arg5[%arg8, %5] : memref<?x1100xf64>
        %8 = affine.apply #map2(%arg9)
        %9 = affine.load %arg5[%arg8, %8] : memref<?x1100xf64>
        %10 = arith.mulf %9, %arg4 : f64
        affine.store %10, %arg5[%arg8, %8] : memref<?x1100xf64>
        %11 = affine.apply #map3(%arg9)
        %12 = affine.load %arg5[%arg8, %11] : memref<?x1100xf64>
        %13 = arith.mulf %12, %arg4 : f64
        affine.store %13, %arg5[%arg8, %11] : memref<?x1100xf64>
      }
      affine.for %arg9 = #map()[%0] to %0 {
        %3 = affine.load %arg5[%arg8, %arg9] : memref<?x1100xf64>
        %4 = arith.mulf %3, %arg4 : f64
        affine.store %4, %arg5[%arg8, %arg9] : memref<?x1100xf64>
      }
      affine.for %arg9 = 0 to %1 {
        affine.for %arg10 = 0 to #map()[%0] step 4 {
          %3 = affine.load %arg6[%arg8, %arg9] : memref<?x1200xf64>
          %4 = arith.mulf %arg3, %3 : f64
          %5 = affine.load %arg7[%arg9, %arg10] : memref<?x1100xf64>
          %6 = arith.mulf %4, %5 : f64
          %7 = affine.load %arg5[%arg8, %arg10] : memref<?x1100xf64>
          %8 = arith.addf %7, %6 : f64
          affine.store %8, %arg5[%arg8, %arg10] : memref<?x1100xf64>
          %9 = affine.apply #map1(%arg10)
          %10 = affine.load %arg6[%arg8, %arg9] : memref<?x1200xf64>
          %11 = arith.mulf %arg3, %10 : f64
          %12 = affine.load %arg7[%arg9, %9] : memref<?x1100xf64>
          %13 = arith.mulf %11, %12 : f64
          %14 = affine.load %arg5[%arg8, %9] : memref<?x1100xf64>
          %15 = arith.addf %14, %13 : f64
          affine.store %15, %arg5[%arg8, %9] : memref<?x1100xf64>
          %16 = affine.apply #map2(%arg10)
          %17 = affine.load %arg6[%arg8, %arg9] : memref<?x1200xf64>
          %18 = arith.mulf %arg3, %17 : f64
          %19 = affine.load %arg7[%arg9, %16] : memref<?x1100xf64>
          %20 = arith.mulf %18, %19 : f64
          %21 = affine.load %arg5[%arg8, %16] : memref<?x1100xf64>
          %22 = arith.addf %21, %20 : f64
          affine.store %22, %arg5[%arg8, %16] : memref<?x1100xf64>
          %23 = affine.apply #map3(%arg10)
          %24 = affine.load %arg6[%arg8, %arg9] : memref<?x1200xf64>
          %25 = arith.mulf %arg3, %24 : f64
          %26 = affine.load %arg7[%arg9, %23] : memref<?x1100xf64>
          %27 = arith.mulf %25, %26 : f64
          %28 = affine.load %arg5[%arg8, %23] : memref<?x1100xf64>
          %29 = arith.addf %28, %27 : f64
          affine.store %29, %arg5[%arg8, %23] : memref<?x1100xf64>
        }
        affine.for %arg10 = #map()[%0] to %0 {
          %3 = affine.load %arg6[%arg8, %arg9] : memref<?x1200xf64>
          %4 = arith.mulf %arg3, %3 : f64
          %5 = affine.load %arg7[%arg9, %arg10] : memref<?x1100xf64>
          %6 = arith.mulf %4, %5 : f64
          %7 = affine.load %arg5[%arg8, %arg10] : memref<?x1100xf64>
          %8 = arith.addf %7, %6 : f64
          affine.store %8, %arg5[%arg8, %arg10] : memref<?x1100xf64>
        }
      }
    }
    return
  }
}
