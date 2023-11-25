  func.func @multiply_matrices(%arg0: i32, %arg1: i32, %arg2: memref<?x?xi32>, %arg3: i32, %arg4: i32, %arg5: memref<?x?xi32>, %arg6: memref<?x?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.index_cast %arg4 : i32 to index
    %3 = arith.index_cast %arg1 : i32 to index
    affine.for %arg7 = %c0 to %1 {
      affine.for %arg8 = %c0 to %2 {
        memref.store %c0_i32, %arg6[%arg7, %arg8] : memref<?x?xi32>
        affine.for %arg9 = %c0 to %3 {
          %4 = memref.load %arg2[%arg7, %arg9] : memref<?x?xi32>
          %5 = memref.load %arg5[%arg9, %arg8] : memref<?x?xi32>
          %6 = arith.muli %4, %5 : i32
          %7 = memref.load %arg6[%arg7, %arg8] : memref<?x?xi32>
          %8 = arith.addi %7, %6 : i32
          memref.store %8, %arg6[%arg7, %arg8] : memref<?x?xi32>
        }
      }
    }
    return
  }
