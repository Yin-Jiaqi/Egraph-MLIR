module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @multiply_matrices(%arg0: i32, %arg1: i32, %arg2: memref<?x?xi32>, %arg3: i32, %arg4: i32, %arg5: memref<?x?xi32>, %arg6: memref<?x?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {

    return
  }
}





    scf.for %arg7 = %c0 to %0 step %c1 {
      scf.for %arg8 = %c0 to %1 step %c1 {
        memref.store %c0_i32, %arg6[%arg7, %arg8] : memref<?x?xi32>
        scf.for %arg9 = %c0 to %2 step %c1 {
          %3 = memref.load %arg2[%arg7, %arg9] : memref<?x?xi32>
          %4 = memref.load %arg5[%arg9, %arg8] : memref<?x?xi32>
          %6 = memref.load %arg6[%arg7, %arg8] : memref<?x?xi32>
          memref.store %7, %arg6[%arg7, %arg8] : memref<?x?xi32>
        }
      }
    }