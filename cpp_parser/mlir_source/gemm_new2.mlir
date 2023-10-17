  func.func @multiply_matrices(%arg0: i32, %arg1: i32, %arg2: memref<?x?xi32>, %arg3: i32, %arg4: i32, %arg5: memref<?x?xi32>, %arg6: memref<?x?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = arith.index_cast %arg1 : i32 to index
    scf.for %arg7 = %c0 to %0 step %c1 {
      scf.for %arg8 = %c0 to %1 step %c1 {
        memref.store %c0_i32, %arg6[%arg7, %arg8] : memref<?x?xi32>
        scf.for %arg9 = %c0 to %2 step %c1 {
          %3 = memref.load %arg2[%arg7, %arg9] : memref<?x?xi32>
          %4 = memref.load %arg5[%arg9, %arg8] : memref<?x?xi32>
          %5 = arith.muli %3, %4 : i32
          %6 = memref.load %arg6[%arg7, %arg8] : memref<?x?xi32>
          %7 = arith.addi %6, %5 : i32
          memref.store %7, %arg6[%arg7, %arg8] : memref<?x?xi32>
        }
      }
    }
    return
  }