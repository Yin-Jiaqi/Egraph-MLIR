#![recursion_limit = "512"]

pub use egg::*;


define_language! {
    pub enum SimpleLanguage {
        NumI32(i32),
        NumI64(i64),
        Symbol(Symbol),
        "+" = Add(Vec<Id>),
        "*" = Mul(Vec<Id>), 
        "arith_addi_i32_0" = ArithAddiI32_0([Id; 2]),
        "arith_addf_f32_0" = ArithAddfF32_0([Id; 2]),
        "arith_addi_i32_1" = ArithAddiI32_1([Id; 4]),
        "arith_addf_f32_1" = ArithAddfF32_1([Id; 4]),
        "arith_addi_i32_2" = ArithAddiI32_2([Id; 6]),
        "arith_addf_f32_2" = ArithAddfF32_2([Id; 6]),
        "arith_muli_i32_0" = ArithMuliI32_0([Id; 2]),
        "arith_mulf_f32_0" = ArithMulfF32_0([Id; 2]),
        "arith_muli_i32_1" = ArithMuliI32_1([Id; 4]),
        "arith_mulf_f32_1" = ArithMulfF32_1([Id; 4]),
        "arith_muli_i32_2" = ArithMuliI32_2([Id; 6]),
        "arith_mulf_f32_2" = ArithMulfF32_2([Id; 6]),
        "arith_addi_i64_0" = ArithAddiI64_0([Id; 2]),
        "arith_addf_f64_0" = ArithAddfF64_0([Id; 2]),
        "arith_addi_i64_1" = ArithAddiI64_1([Id; 4]),
        "arith_addf_f64_1" = ArithAddfF64_1([Id; 4]),
        "arith_addi_i64_2" = ArithAddiI64_2([Id; 6]),
        "arith_addf_f64_2" = ArithAddfF64_2([Id; 6]),
        "arith_muli_i64_0" = ArithMuliI64_0([Id; 2]),
        "arith_mulf_f64_0" = ArithMulfF64_0([Id; 2]),
        "arith_muli_i64_1" = ArithMuliI64_1([Id; 4]),
        "arith_mulf_f64_1" = ArithMulfF64_1([Id; 4]),
        "arith_muli_i64_2" = ArithMuliI64_2([Id; 6]),
        "arith_mulf_f64_2" = ArithMulfF64_2([Id; 6]),
        "arith_constant_i32_0" = ArithConstantI32_0([Id; 1]),
        "arith_constant_index_0" = ArithConstantIndex_0([Id; 1]),
        "arith_indexcast_i32index_0" = ArithIndexCastI32index_0([Id; 1]),
        "memref_load_i32_0" = MemrefLoadI32_0([Id; 1]),
        "memref_load_i32_1" = MemrefLoadI32_1([Id; 2]),
        "memref_load_i32_2" = MemrefLoadI32_2([Id; 3]),
        "memref_load_i32_3" = MemrefLoadI32_3([Id; 4]),
        "memref_store_i32_0" = MemrefStoreI32_0([Id; 2]),
        "memref_store_i32_1" = MemrefStoreI32_1([Id; 3]),
        "memref_store_i32_2" = MemrefStoreI32_2([Id; 4]),
        "memref_store_i32_3" = MemrefStoreI32_3([Id; 5]),
        "affine_load_i32_0" = AffineLoadI32_0([Id; 1]),
        "affine_load_i32_1" = AffineLoadI32_1([Id; 2]),
        "affine_load_i32_2" = AffineLoadI32_2([Id; 3]),
        "affine_load_i32_3" = AffineLoadI32_3([Id; 4]),
        "affine_store_i32_0" = AffineStoreI32_0([Id; 2]),
        "affine_store_i32_1" = AffineStoreI32_1([Id; 3]),
        "affine_store_i32_2" = AffineStoreI32_2([Id; 4]),
        "affine_store_i32_3" = AffineStoreI32_3([Id; 5]),
        "memref_load_f64_0" = MemrefLoadF64_0([Id; 1]),
        "memref_load_f64_1" = MemrefLoadF64_1([Id; 2]),
        "memref_load_f64_2" = MemrefLoadF64_2([Id; 3]),
        "memref_load_f64_3" = MemrefLoadF64_3([Id; 4]),
        "memref_store_f64_0" = MemrefStoreF64_0([Id; 2]),
        "memref_store_f64_1" = MemrefStoreF64_1([Id; 3]),
        "memref_store_f64_2" = MemrefStoreF64_2([Id; 4]),
        "memref_store_f64_3" = MemrefStoreF64_3([Id; 5]),
        "affine_load_f64_0" = AffineLoadF64_0([Id; 1]),
        "affine_load_f64_1" = AffineLoadF64_1([Id; 2]),
        "affine_load_f64_2" = AffineLoadF64_2([Id; 3]),
        "affine_load_f64_3" = AffineLoadF64_3([Id; 4]),
        "affine_store_f64_0" = AffineStoreF64_0([Id; 2]),
        "affine_store_f64_1" = AffineStoreF64_1([Id; 3]),
        "affine_store_f64_2" = AffineStoreF64_2([Id; 4]),
        "affine_store_f64_3" = AffineStoreF64_3([Id; 5]),
        "affine_forvalue_none_0" = AffineForvalueNone_0([Id; 3]),
        "affine_forcontrol_none_0" = AffineForcontrolNone_0([Id; 2]),
        "scf_forvalue_none_0" = ScfForvalueNone_0([Id; 3]),
        "scf_forcontrol_none_0" = ScfForcontrolNone_0([Id; 2]),
        "func_func_none" = FuncFuncNone(Vec<Id>),
        "block_block_none" = BlockBlockNone(Vec<Id>),
        // "func_func_none_1" = FuncFuncNone_1([Id; 1]),
        // "func_func_none_2" = FuncFuncNone_2([Id; 2]),
        // "func_func_none_3" = FuncFuncNone_3([Id; 3]),
        // "func_func_none_4" = FuncFuncNone_4([Id; 4]),
        // "func_func_none_5" = FuncFuncNone_5([Id; 5]),
        // "func_func_none_6" = FuncFuncNone_6([Id; 6]),
        // "func_func_none_7" = FuncFuncNone_7([Id; 7]),
        // "func_func_none_8" = FuncFuncNone_8([Id; 8]),
        // "func_func_none_9" = FuncFuncNone_9([Id; 9]),
        // "func_func_none_10" = FuncFuncNone_10([Id; 10]),
        // "block_block_none_0" = BlockBlockNone_0([Id; 0]),
        // "block_block_none_1" = BlockBlockNone_1([Id; 1]),
        // "block_block_none_2" = BlockBlockNone_2([Id; 2]),
        // "block_block_none_3" = BlockBlockNone_3([Id; 3]),
        // "block_block_none_4" = BlockBlockNone_4([Id; 4]),
        // "block_block_none_5" = BlockBlockNone_5([Id; 5]),
        // "block_block_none_6" = BlockBlockNone_6([Id; 6]),
        // "block_block_none_7" = BlockBlockNone_7([Id; 7]),
        // "block_block_none_8" = BlockBlockNone_8([Id; 8]),
        // "block_block_none_9" = BlockBlockNone_9([Id; 9]),
        // "block_block_none_10" = BlockBlockNone_10([Id; 10]),
        // "block_block_none_11" = BlockBlockNone_11([Id; 11]),
    }
}




pub fn convert_to_simple_language_enum(in_id: Vec<Id>, stripped_name: &str) -> Option<SimpleLanguage> {
    let language_enum = match stripped_name {
        "arith_addi_i32_0" => {
            let arr: [Id; 2] = [in_id[0], in_id[1]];
            Some(SimpleLanguage::ArithAddiI32_0(arr))
        },
        "arith_addf_f32_0" => {
            let arr: [Id; 2] = [in_id[0], in_id[1]];
            Some(SimpleLanguage::ArithAddfF32_0(arr))
        },
        "arith_addi_i32_1" => {
            let arr: [Id; 4] = [in_id[0], in_id[1], in_id[2], in_id[3]];
            Some(SimpleLanguage::ArithAddiI32_1(arr))
        },
        "arith_addf_f32_1" => {
            let arr: [Id; 4] = [in_id[0], in_id[1], in_id[2], in_id[3]];
            Some(SimpleLanguage::ArithAddfF32_1(arr))
        },
        "arith_addi_i32_2" => {
            let arr: [Id; 6] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4], in_id[5]];
            Some(SimpleLanguage::ArithAddiI32_2(arr))
        },
        "arith_addf_f32_2" => {
            let arr: [Id; 6] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4], in_id[5]];
            Some(SimpleLanguage::ArithAddfF32_2(arr))
        },
        "arith_muli_i32_0" => {
            let arr: [Id; 2] = [in_id[0], in_id[1]];
            Some(SimpleLanguage::ArithMuliI32_0(arr))
        },
        "arith_mulf_f32_0" => {
            let arr: [Id; 2] = [in_id[0], in_id[1]];
            Some(SimpleLanguage::ArithMulfF32_0(arr))
        },
        "arith_muli_i32_1" => {
            let arr: [Id; 4] = [in_id[0], in_id[1], in_id[2], in_id[3]];
            Some(SimpleLanguage::ArithMuliI32_1(arr))
        },
        "arith_mulf_f32_1" => {
            let arr: [Id; 4] = [in_id[0], in_id[1], in_id[2], in_id[3]];
            Some(SimpleLanguage::ArithMulfF32_1(arr))
        },
        "arith_muli_i32_2" => {
            let arr: [Id; 6] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4], in_id[5]];
            Some(SimpleLanguage::ArithMuliI32_2(arr))
        },
        "arith_mulf_f32_2" => {
            let arr: [Id; 6] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4], in_id[5]];
            Some(SimpleLanguage::ArithMulfF32_2(arr))
        },
        "arith_addi_i64_0" => {
            let arr: [Id; 2] = [in_id[0], in_id[1]];
            Some(SimpleLanguage::ArithAddiI64_0(arr))
        },
        "arith_addf_f64_0" => {
            let arr: [Id; 2] = [in_id[0], in_id[1]];
            Some(SimpleLanguage::ArithAddfF64_0(arr))
        },
        "arith_addi_i64_1" => {
            let arr: [Id; 4] = [in_id[0], in_id[1], in_id[2], in_id[3]];
            Some(SimpleLanguage::ArithAddiI64_1(arr))
        },
        "arith_addf_f64_1" => {
            let arr: [Id; 4] = [in_id[0], in_id[1], in_id[2], in_id[3]];
            Some(SimpleLanguage::ArithAddfF64_1(arr))
        },
        "arith_addi_i64_2" => {
            let arr: [Id; 6] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4], in_id[5]];
            Some(SimpleLanguage::ArithAddiI64_2(arr))
        },
        "arith_addf_f64_2" => {
            let arr: [Id; 6] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4], in_id[5]];
            Some(SimpleLanguage::ArithAddfF64_2(arr))
        },
        "arith_muli_i64_0" => {
            let arr: [Id; 2] = [in_id[0], in_id[1]];
            Some(SimpleLanguage::ArithMuliI64_0(arr))
        },
        "arith_mulf_f64_0" => {
            let arr: [Id; 2] = [in_id[0], in_id[1]];
            Some(SimpleLanguage::ArithMulfF64_0(arr))
        },
        "arith_muli_i64_1" => {
            let arr: [Id; 4] = [in_id[0], in_id[1], in_id[2], in_id[3]];
            Some(SimpleLanguage::ArithMuliI64_1(arr))
        },
        "arith_mulf_f64_1" => {
            let arr: [Id; 4] = [in_id[0], in_id[1], in_id[2], in_id[3]];
            Some(SimpleLanguage::ArithMulfF64_1(arr))
        },
        "arith_muli_i64_2" => {
            let arr: [Id; 6] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4], in_id[5]];
            Some(SimpleLanguage::ArithMuliI64_2(arr))
        },
        "arith_mulf_f64_2" => {
            let arr: [Id; 6] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4], in_id[5]];
            Some(SimpleLanguage::ArithMulfF64_2(arr))
        },
        "arith_constant_i32_0" => {
            let arr: [Id; 1] = [in_id[0]];
            Some(SimpleLanguage::ArithConstantI32_0(arr))
        },
        "arith_constant_index_0" => {
            let arr: [Id; 1] = [in_id[0]];
            Some(SimpleLanguage::ArithConstantIndex_0(arr))
        },
        "arith_indexcast_i32index_0" => {
            let arr: [Id; 1] = [in_id[0]];
            Some(SimpleLanguage::ArithIndexCastI32index_0(arr))
        },
        "memref_load_i32_0" => {
            let arr: [Id; 1] = [in_id[0]];
            Some(SimpleLanguage::MemrefLoadI32_0(arr))
        },
        "memref_load_i32_1" => {
            let arr: [Id; 2] = [in_id[0], in_id[1]];
            Some(SimpleLanguage::MemrefLoadI32_1(arr))
        },
        "memref_load_i32_2" => {
            let arr: [Id; 3] = [in_id[0], in_id[1], in_id[2]];
            Some(SimpleLanguage::MemrefLoadI32_2(arr))
        },
        "memref_load_i32_3" => {
            let arr: [Id; 4] = [in_id[0], in_id[1], in_id[2], in_id[3]];
            Some(SimpleLanguage::MemrefLoadI32_3(arr))
        },
        "memref_store_i32_0" => {
            let arr: [Id; 2] = [in_id[0], in_id[1]];
            Some(SimpleLanguage::MemrefStoreI32_0(arr))
        },
        "memref_store_i32_1" => {
            let arr: [Id; 3] = [in_id[0], in_id[1], in_id[2]];
            Some(SimpleLanguage::MemrefStoreI32_1(arr))
        },
        "memref_store_i32_2" => {
            let arr: [Id; 4] = [in_id[0], in_id[1], in_id[2], in_id[3]];
            Some(SimpleLanguage::MemrefStoreI32_2(arr))
        },
        "memref_store_i32_3" => {
            let arr: [Id; 5] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4]];
            Some(SimpleLanguage::MemrefStoreI32_3(arr))
        },
        "affine_load_i32_0" => {
            let arr: [Id; 1] = [in_id[0]];
            Some(SimpleLanguage::AffineLoadI32_0(arr))
        },
        "affine_load_i32_1" => {
            let arr: [Id; 2] = [in_id[0], in_id[1]];
            Some(SimpleLanguage::AffineLoadI32_1(arr))
        },
        "affine_load_i32_2" => {
            let arr: [Id; 3] = [in_id[0], in_id[1], in_id[2]];
            Some(SimpleLanguage::AffineLoadI32_2(arr))
        },
        "affine_load_i32_3" => {
            let arr: [Id; 4] = [in_id[0], in_id[1], in_id[2], in_id[3]];
            Some(SimpleLanguage::AffineLoadI32_3(arr))
        },
        "affine_store_i32_0" => {
            let arr: [Id; 2] = [in_id[0], in_id[1]];
            Some(SimpleLanguage::AffineStoreI32_0(arr))
        },
        "affine_store_i32_1" => {
            let arr: [Id; 3] = [in_id[0], in_id[1], in_id[2]];
            Some(SimpleLanguage::AffineStoreI32_1(arr))
        },
        "affine_store_i32_2" => {
            let arr: [Id; 4] = [in_id[0], in_id[1], in_id[2], in_id[3]];
            Some(SimpleLanguage::AffineStoreI32_2(arr))
        },
        "affine_store_i32_3" => {
            let arr: [Id; 5] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4]];
            Some(SimpleLanguage::AffineStoreI32_3(arr))
        },
        "memref_load_f64_0" => {
            let arr: [Id; 1] = [in_id[0]];
            Some(SimpleLanguage::MemrefLoadF64_0(arr))
        },
        "memref_load_f64_1" => {
            let arr: [Id; 2] = [in_id[0], in_id[1]];
            Some(SimpleLanguage::MemrefLoadF64_1(arr))
        },
        "memref_load_f64_2" => {
            let arr: [Id; 3] = [in_id[0], in_id[1], in_id[2]];
            Some(SimpleLanguage::MemrefLoadF64_2(arr))
        },
        "memref_load_f64_3" => {
            let arr: [Id; 4] = [in_id[0], in_id[1], in_id[2], in_id[3]];
            Some(SimpleLanguage::MemrefLoadF64_3(arr))
        },
        "memref_store_f64_0" => {
            let arr: [Id; 2] = [in_id[0], in_id[1]];
            Some(SimpleLanguage::MemrefStoreF64_0(arr))
        },
        "memref_store_f64_1" => {
            let arr: [Id; 3] = [in_id[0], in_id[1], in_id[2]];
            Some(SimpleLanguage::MemrefStoreF64_1(arr))
        },
        "memref_store_f64_2" => {
            let arr: [Id; 4] = [in_id[0], in_id[1], in_id[2], in_id[3]];
            Some(SimpleLanguage::MemrefStoreF64_2(arr))
        },
        "memref_store_f64_3" => {
            let arr: [Id; 5] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4]];
            Some(SimpleLanguage::MemrefStoreF64_3(arr))
        },
        "affine_load_f64_0" => {
            let arr: [Id; 1] = [in_id[0]];
            Some(SimpleLanguage::AffineLoadF64_0(arr))
        },
        "affine_load_f64_1" => {
            let arr: [Id; 2] = [in_id[0], in_id[1]];
            Some(SimpleLanguage::AffineLoadF64_1(arr))
        },
        "affine_load_f64_2" => {
            let arr: [Id; 3] = [in_id[0], in_id[1], in_id[2]];
            Some(SimpleLanguage::AffineLoadF64_2(arr))
        },
        "affine_load_f64_3" => {
            let arr: [Id; 4] = [in_id[0], in_id[1], in_id[2], in_id[3]];
            Some(SimpleLanguage::AffineLoadF64_3(arr))
        },
        "affine_store_f64_0" => {
            let arr: [Id; 2] = [in_id[0], in_id[1]];
            Some(SimpleLanguage::AffineStoreF64_0(arr))
        },
        "affine_store_f64_1" => {
            let arr: [Id; 3] = [in_id[0], in_id[1], in_id[2]];
            Some(SimpleLanguage::AffineStoreF64_1(arr))
        },
        "affine_store_f64_2" => {
            let arr: [Id; 4] = [in_id[0], in_id[1], in_id[2], in_id[3]];
            Some(SimpleLanguage::AffineStoreF64_2(arr))
        },
        "affine_store_f64_3" => {
            let arr: [Id; 5] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4]];
            Some(SimpleLanguage::AffineStoreF64_3(arr))
        },
        "affine_forvalue_none_0" => {
            let arr: [Id; 3] = [in_id[0], in_id[1], in_id[2]];
            Some(SimpleLanguage::AffineForvalueNone_0(arr))
        },
        "affine_forcontrol_none_0" => {
            let arr: [Id; 2] = [in_id[0], in_id[1]];
            Some(SimpleLanguage::AffineForcontrolNone_0(arr))
        },
        "scf_forvalue_none_0" => {
            let arr: [Id; 3] = [in_id[0], in_id[1], in_id[2]];
            Some(SimpleLanguage::ScfForvalueNone_0(arr))
        },
        "scf_forcontrol_none_0" => {
            let arr: [Id; 2] = [in_id[0], in_id[1]];
            Some(SimpleLanguage::ScfForcontrolNone_0(arr))
        },
        "func_func_none" => {
            Some(SimpleLanguage::FuncFuncNone(in_id))
        },
        "block_block_none" => {
            Some(SimpleLanguage::BlockBlockNone(in_id))
        },
        // "func_func_none_1" => {
        //     let arr: [Id; 1] = [in_id[0]];
        //     Some(SimpleLanguage::FuncFuncNone_1(arr))
        // },
        // "func_func_none_2" => {
        //     let arr: [Id; 2] = [in_id[0], in_id[1]];
        //     Some(SimpleLanguage::FuncFuncNone_2(arr))
        // },
        // "func_func_none_3" => {
        //     let arr: [Id; 3] = [in_id[0], in_id[1], in_id[2]];
        //     Some(SimpleLanguage::FuncFuncNone_3(arr))
        // },
        // "func_func_none_4" => {
        //     let arr: [Id; 4] = [in_id[0], in_id[1], in_id[2], in_id[3]];
        //     Some(SimpleLanguage::FuncFuncNone_4(arr))
        // },
        // "func_func_none_5" => {
        //     let arr: [Id; 5] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4]];
        //     Some(SimpleLanguage::FuncFuncNone_5(arr))
        // },
        // "func_func_none_6" => {
        //     let arr: [Id; 6] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4], in_id[5]];
        //     Some(SimpleLanguage::FuncFuncNone_6(arr))
        // },
        // "func_func_none_7" => {
        //     let arr: [Id; 7] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4], in_id[5], in_id[6]];
        //     Some(SimpleLanguage::FuncFuncNone_7(arr))
        // },
        // "func_func_none_8" => {
        //     let arr: [Id; 8] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4], in_id[5], in_id[6], in_id[7]];
        //     Some(SimpleLanguage::FuncFuncNone_8(arr))
        // },
        // "func_func_none_9" => {
        //     let arr: [Id; 9] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4], in_id[5], in_id[6], in_id[7], in_id[8]];
        //     Some(SimpleLanguage::FuncFuncNone_9(arr))
        // },
        // "func_func_none_10" => {
        //     let arr: [Id; 10] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4], in_id[5], in_id[6], in_id[7], in_id[8], in_id[9]];
        //     Some(SimpleLanguage::FuncFuncNone_10(arr))
        // },
        // "block_block_none_0" => {
        //     let arr: [Id; 0] = [];
        //     Some(SimpleLanguage::BlockBlockNone_0(arr))
        // },
        // "block_block_none_1" => {
        //     let arr: [Id; 1] = [in_id[0]];
        //     Some(SimpleLanguage::BlockBlockNone_1(arr))
        // },
        // "block_block_none_2" => {
        //     let arr: [Id; 2] = [in_id[0], in_id[1]];
        //     Some(SimpleLanguage::BlockBlockNone_2(arr))
        // },
        // "block_block_none_3" => {
        //     let arr: [Id; 3] = [in_id[0], in_id[1], in_id[2]];
        //     Some(SimpleLanguage::BlockBlockNone_3(arr))
        // },
        // "block_block_none_4" => {
        //     let arr: [Id; 4] = [in_id[0], in_id[1], in_id[2], in_id[3]];
        //     Some(SimpleLanguage::BlockBlockNone_4(arr))
        // },
        // "block_block_none_5" => {
        //     let arr: [Id; 5] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4]];
        //     Some(SimpleLanguage::BlockBlockNone_5(arr))
        // },
        // "block_block_none_6" => {
        //     let arr: [Id; 6] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4], in_id[5]];
        //     Some(SimpleLanguage::BlockBlockNone_6(arr))
        // },
        // "block_block_none_7" => {
        //     let arr: [Id; 7] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4], in_id[5], in_id[6]];
        //     Some(SimpleLanguage::BlockBlockNone_7(arr))
        // },
        // "block_block_none_8" => {
        //     let arr: [Id; 8] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4], in_id[5], in_id[6], in_id[7]];
        //     Some(SimpleLanguage::BlockBlockNone_8(arr))
        // },
        // "block_block_none_9" => {
        //     let arr: [Id; 9] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4], in_id[5], in_id[6], in_id[7], in_id[8]];
        //     Some(SimpleLanguage::BlockBlockNone_9(arr))
        // },
        // "block_block_none_10" => {
        //     let arr: [Id; 10] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4], in_id[5], in_id[6], in_id[7], in_id[8], in_id[9]];
        //     Some(SimpleLanguage::BlockBlockNone_10(arr))
        // },
        // "block_block_none_11" => {
        //     let arr: [Id; 11] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4], in_id[5], in_id[6], in_id[7], in_id[8], in_id[9], in_id[10]];
        //     Some(SimpleLanguage::BlockBlockNone_11(arr))
        // },
        _ => None,
    };
    language_enum
}

