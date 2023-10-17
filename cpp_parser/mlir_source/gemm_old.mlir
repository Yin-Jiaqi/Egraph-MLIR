  func.func @multiply_matrices(%arg0: i32, %arg1: i32, %arg2: memref<?x?xi32>, %arg3: i32, %arg4: i32, %arg5: memref<?x?xi32>, %arg6: memref<?x?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi ne, %arg1, %arg3 : i32
    scf.if %0 {
      %1 = llvm.mlir.addressof @str0 : !llvm.ptr<array<130 x i8>>
      %2 = llvm.getelementptr %1[0, 0] : (!llvm.ptr<array<130 x i8>>) -> !llvm.ptr<i8>
      %3 = llvm.call @printf(%2) : (!llvm.ptr<i8>) -> i32
    } else {
      %1 = arith.index_cast %arg0 : i32 to index
      %2 = arith.index_cast %arg4 : i32 to index
      %3 = arith.index_cast %arg1 : i32 to index
      scf.for %arg7 = %c0 to %1 step %c1 {
        scf.for %arg8 = %c0 to %2 step %c1 {
          memref.store %c0_i32, %arg6[%arg7, %arg8] : memref<?x?xi32>
          scf.for %arg9 = %c0 to %3 step %c1 {
            %4 = memref.load %arg2[%arg7, %arg9] : memref<?x?xi32>
            %5 = memref.load %arg5[%arg9, %arg8] : memref<?x?xi32>
            %6 = arith.muli %4, %5 : i32
            %7 = memref.load %arg6[%arg7, %arg8] : memref<?x?xi32>
            %8 = arith.addi %7, %6 : i32
            memref.store %8, %arg6[%arg7, %arg8] : memref<?x?xi32>
          }
        }
      }
    }
    return
  }
