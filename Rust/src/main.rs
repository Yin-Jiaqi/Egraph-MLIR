use Mlir_Egg::parse_graph;
use regex::Regex;
use std::collections::HashMap;
use std::fs;
use egg::*;

define_language! {
    enum SimpleLanguage {
        NumI32(i32),
        NumI64(i64),
        // Num_f32(f32),
        // Num_f64(f64),
        Symbol(Symbol),
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
        "arith_constant_i32_0" = ArithConstantI32_0([Id; 1]),
        "arith_constant_index_0" = ArithConstantIndex_0([Id; 1]),
        "arith_index_cast_0" = ArithIndexCastI32index_0([Id; 1]),
        "memref_load_i32_0" = MemrefLoadI32_0([Id; 1]),
        "memref_load_i32_1" = MemrefLoadI32_1([Id; 2]),
        "memref_load_i32_2" = MemrefLoadI32_2([Id; 3]),
        "memref_load_i32_3" = MemrefLoadI32_3([Id; 4]),
        "memref_store_i32_0" = MemrefStoreI32_0([Id; 2]),
        "memref_store_i32_1" = MemrefStoreI32_1([Id; 3]),
        "memref_store_i32_2" = MemrefStoreI32_2([Id; 4]),
        "memref_store_i32_3" = MemrefStoreI32_3([Id; 5]),
        "func_func_none_0"=FuncFuncNone_0([Id; 0]),
        "func_func_none_1"=FuncFuncNone_1([Id; 1]),
        "func_func_none_2"=FuncFuncNone_2([Id; 2]),
        "func_func_none_3"=FuncFuncNone_3([Id; 3]),
        "func_func_none_4"=FuncFuncNone_4([Id; 4]),
        "func_func_none_5"=FuncFuncNone_5([Id; 5]),
        "func_func_none_6"=FuncFuncNone_6([Id; 6]),
        "func_func_none_7"=FuncFuncNone_7([Id; 7]),
        "func_func_none_8"=FuncFuncNone_8([Id; 8]),
    }
}

fn main() {
    let input = fs::read_to_string("output.txt").expect("Failed to read file");
    let (vertices, edges, in_pairs, out_pairs) = parse_graph(&input);
    let mut expr: RecExpr<SimpleLanguage> = RecExpr::default();
    let mut edge_map = HashMap::new();
    let mut node_map = HashMap::new();
    
    for e in edges {
        let temp_id = expr.add(SimpleLanguage::Symbol(e.clone().into()));
        edge_map.insert(e, temp_id);
    }

    for (src, des) in in_pairs {
        let mut in_id = Vec::new();
        for e in des {
            in_id.push(*edge_map.get(&e).unwrap());
        }
        
        let name: Vec<&str> = src.rsplitn(2, '_').collect();
        let stripped_name = name.last().unwrap();
        
        let language_enum = match *stripped_name {
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
            "arith_constant_i32_0" => {
                let arr: [Id; 1] = [in_id[0]];
                Some(SimpleLanguage::ArithConstantI32_0(arr))
            },
            "arith_constant_index_0" => {
                let arr: [Id; 1] = [in_id[0]];
                Some(SimpleLanguage::ArithConstantIndex_0(arr))
            },
            "arith_index_cast_0" => {
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
            "func_func_none_0" => {
                let arr: [Id; 0] = [];
                Some(SimpleLanguage::FuncFuncNone_0(arr))
            },
            "func_func_none_1" => {
                let arr: [Id; 1] = [in_id[0]];
                Some(SimpleLanguage::FuncFuncNone_1(arr))
            },
            "func_func_none_2" => {
                let arr: [Id; 2] = [in_id[0], in_id[1]];
                Some(SimpleLanguage::FuncFuncNone_2(arr))
            },
            "func_func_none_3" => {
                let arr: [Id; 3] = [in_id[0], in_id[1], in_id[2]];
                Some(SimpleLanguage::FuncFuncNone_3(arr))
            },
            "func_func_none_4" => {
                let arr: [Id; 4] = [in_id[0], in_id[1], in_id[2], in_id[3]];
                Some(SimpleLanguage::FuncFuncNone_4(arr))
            },
            "func_func_none_5" => {
                let arr: [Id; 5] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4]];
                Some(SimpleLanguage::FuncFuncNone_5(arr))
            },
            "func_func_none_6" => {
                let arr: [Id; 6] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4], in_id[5]];
                Some(SimpleLanguage::FuncFuncNone_6(arr))
            },
            "func_func_none_7" => {
                let arr: [Id; 7] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4], in_id[5], in_id[6]];
                Some(SimpleLanguage::FuncFuncNone_7(arr))
            },
            "func_func_none_8" => {
                let arr: [Id; 8] = [in_id[0], in_id[1], in_id[2], in_id[3], in_id[4], in_id[5], in_id[6], in_id[7]];
                Some(SimpleLanguage::FuncFuncNone_8(arr))
            },
            _ => None,
        };
        
        
        if let Some(value) = language_enum {
            let temp_id = expr.add(value);
            node_map.insert(src, temp_id);
        } else {
            println!("Unknown enum variant.");
        }
    }
}