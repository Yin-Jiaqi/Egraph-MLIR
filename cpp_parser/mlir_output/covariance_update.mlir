#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_covariance(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x1200xf64>, %arg4: memref<?x1200xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 1.000000e+00 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    affine.for %arg6 = 0 to %1 {
      affine.store %cst_0, %arg5[%arg6] : memref<?xf64>
      affine.for %arg7 = 0 to %0 {
        %2 = affine.load %arg3[%arg7, %arg6] : memref<?x1200xf64>
        %3 = affine.load %arg5[%arg6] : memref<?xf64>
        %4 = arith.addf %3, %2 : f64
        affine.store %4, %arg5[%arg6] : memref<?xf64>
      }
      %5 = affine.load %arg5[%arg6] : memref<?xf64>
      %6 = arith.divf %5, %arg2 : f64
      affine.store %6, %arg5[%arg6] : memref<?xf64>
    }
    affine.for %arg8 = 0 to %0 {
      affine.for %arg9 = 0 to %1 {
        %7 = affine.load %arg5[%arg9] : memref<?xf64>
        %8 = affine.load %arg3[%arg8, %arg9] : memref<?x1200xf64>
        %9 = arith.subf %8, %7 : f64
        affine.store %9, %arg3[%arg8, %arg9] : memref<?x1200xf64>
      }
    }
    %10 = arith.subf %arg2, %cst_1 : f64
    affine.for %arg10 = 0 to %1 {
      affine.for %arg11 = #map(%arg10) to %1 {
        affine.store %cst_0, %arg4[%arg10, %arg11] : memref<?x1200xf64>
        affine.for %arg12 = 0 to %0 {
          %11 = affine.load %arg3[%arg12, %arg10] : memref<?x1200xf64>
          %12 = affine.load %arg3[%arg12, %arg11] : memref<?x1200xf64>
          %13 = arith.mulf %11, %12 : f64
          %14 = affine.load %arg4[%arg10, %arg11] : memref<?x1200xf64>
          %15 = arith.addf %14, %13 : f64
          affine.store %15, %arg4[%arg10, %arg11] : memref<?x1200xf64>
        }
        %16 = affine.load %arg4[%arg10, %arg11] : memref<?x1200xf64>
        %17 = arith.divf %16, %10 : f64
        affine.store %17, %arg4[%arg10, %arg11] : memref<?x1200xf64>
        affine.store %17, %arg4[%arg11, %arg10] : memref<?x1200xf64>
      }
    }
    return
  }
}



#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @kernel_covariance(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?x1200xf64>, %arg4: memref<?x1200xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 1.000000e+00 : f64
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    affine.for %arg6 = 0 to %1 {
      affine.store %cst_0, %arg5[%arg6] : memref<?xf64>
      affine.for %arg7 = 0 to %0 {
        %2 = affine.load %arg3[%arg7, %arg6] : memref<?x1200xf64>
        %3 = affine.load %arg5[%arg6] : memref<?xf64>
        %4 = arith.addf %3, %2 : f64
        affine.store %4, %arg5[%arg6] : memref<?xf64>
      }
      %5 = affine.load %arg5[%arg6] : memref<?xf64>
      %6 = arith.divf %5, %arg2 : f64
      affine.store %6, %arg5[%arg6] : memref<?xf64>
    }
    affine.for %arg8 = 0 to %0 {
      affine.for %arg9 = 0 to %1 {
        %7 = affine.load %arg5[%arg9] : memref<?xf64>
        %8 = affine.load %arg3[%arg8, %arg9] : memref<?x1200xf64>
        %9 = arith.subf %8, %7 : f64
        affine.store %9, %arg3[%arg8, %arg9] : memref<?x1200xf64>
      }
    }
    %10 = arith.subf %arg2, %cst_1 : f64
    affine.for %arg10 = 0 to %1 {
      affine.for %arg11 = #map(%arg10) to %1 {
        affine.store %cst_0, %arg4[%arg10, %arg11] : memref<?x1200xf64>
        affine.for %arg12 = 0 to %0 {
          %11 = affine.load %arg3[%arg12, %arg10] : memref<?x1200xf64>
          %12 = affine.load %arg3[%arg12, %arg11] : memref<?x1200xf64>
          %13 = arith.mulf %11, %12 : f64
          %14 = affine.load %arg4[%arg10, %arg11] : memref<?x1200xf64>
          %15 = arith.addf %14, %13 : f64
          affine.store %15, %arg4[%arg10, %arg11] : memref<?x1200xf64>
        }
        %16 = affine.load %arg4[%arg10, %arg11] : memref<?x1200xf64>
        %17 = arith.divf %16, %10 : f64
        affine.store %17, %arg4[%arg10, %arg11] : memref<?x1200xf64>
        affine.store %17, %arg4[%arg11, %arg10] : memref<?x1200xf64>
      }
    }
    return
  }
}