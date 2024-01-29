mod lib;
use regex::Regex;
use std::collections::HashMap;
use std::fs;
use egg::*;
use Mlir_Egg::{SimpleLanguage, convert_to_simple_language_enum}; // Add other necessary imports
use std::fs::File;
use std::io::Write;

// rewriting rules
fn make_rules() -> Vec<Rewrite<SimpleLanguage, ()>> {
    vec![
        // rewrite!("block-expand"; "(block_block_none ?a ?b)" => "(block_block_none ?a)"),

        // rewrite!("unrolling_0"; "(affine_forcontrol_none_0 (affine_forvalue_none_0 ?op1 ?op2 ?op3) (block_block_none ?op4))" => "(affine_forcontrol_none_0 (affine_forvalue_none_0 ?op1 ?op2 (* 2 ?op3)) (block_block_none ?op4 ?op4))"),
        // rewrite!("unrolling_0"; "(affine_forcontrol_none_0 (affine_forvalue_none_0 ?op1 ?op2 ?op3) (block_block_none ?op4 ?op5 ?op6))" => "(affine_forcontrol_none_0 (affine_forvalue_none_0 ?op1 ?op2 (* 2 ?op3)) (block_block_none ?op4 ?op5 ?op6))"),
        rewrite!("rule_0"; "(affine_forcontrol_none_0 (affine_forvalue_none_0 ?op1 ?op2 ?op3 ?op4) (block_block_none ?op5))" => "(affine_forcontrol_none_0 (affine_forvalue_none_0 ?op1 ?op2 (* 2 ?op3) ?op4) (block_block_none ?op5 (affine_apply_none (affine_forvalue_none_0 ?op1 ?op2 (* 2 ?op3) ?op4) {d0}->{d0+1}) ?op5))"),
        
        // rewrite!("tilining_0"; "(affine_forcontrol_none_0 (affine_forvalue_none_0 ?op1 ?op2 ?op3) (block_block_none ?op4))" => "(affine_forcontrol_none_0 (affine_forvalue_none_0 ?op1 ?op2 64) (block_block_none (affine_forcontrol_none_0 (affine_forvalue_none_0 (affine_forvalue_none_0 ?op1 ?op2 64) (+ (affine_forvalue_none_0 ?op1 ?op2 64) 64) 1) (block_block_none ?op4))))"), 
    ]
}



// Define a function to build an expression tree using the RecExpr from the egg library.
fn build_expr() -> RecExpr<SimpleLanguage> {
    // Read the input from a file and panic if the file cannot be read.
    let input = fs::read_to_string("mlir_source/mlir_output/jacobi_2d.txt").expect("Failed to read file");
    // Parse the graph from the input file into vertices, edges, and input-output pairs.
    // example:
    // vertices: ["Source_3", "Source_0", "Source_1", "Source_9", "Source_2", "Source_7", "Source_8", "Source_5", "Sink", "Source_6", "Source_4", "Arith_IndexCast_I32Index_0_1_5_2", "Affine_Forvalue_None_0_1_6_0", "Arith_IndexCast_I32Index_0_1_3_0", "Arith_IndexCast_I32Index_0_1_4_1", "Affine_Forvalue_None_0_2_12_2", "Affine_Load_F64_2_4_14_1", "Affine_Forvalue_None_0_3_13_3", "Affine_Load_F64_2_4_16_2", "Affine_Load_F64_2_4_18_3", "Arith_Mulf_F64_0_4_15_1", "Arith_Mulf_F64_0_4_17_2", "Arith_Addf_F64_0_4_19_0", "Affine_Store_F64_2_4_20_1", "Block_Block_None_7_0_13_4", "Affine_Forcontrol_None_0_3_13_3", "Block_Block_None_1_0_12_3", "Affine_Forcontrol_None_0_2_12_2", "Affine_Forvalue_None_0_2_7_1", "Affine_Load_F64_2_3_8_0", "Arith_Mulf_F64_0_3_9_0", "Affine_Store_F64_2_3_10_0", "Block_Block_None_3_0_7_2", "Affine_Forcontrol_None_0_2_7_1", "Block_Block_None_2_0_6_1", "Affine_Forcontrol_None_0_1_6_0", "Block_Block_None_4_0_2_0", "Func_Func_None_9_0_2_0"]
    // edges: ["%arg0_I32_0_2", "%arg1_I32_0_2", "%arg2_I32_0_2", "%arg3_F64_0_2", "%arg4_F64_0_2", "%arg5_F64_2_2", "%arg6_F64_2_2", "%arg7_F64_2_2", "%0_I32Index_0_3", "%1_I32Index_0_4", "%2_I32Index_0_5", "%arg8_None_0_6", "0_None_0_-1", "1_None_0_-1", "Pseudo/0_None_0_6", "%arg9/0_None_0_7", "Pseudo/1_None_0_7", "%3/0_F64_2_8", "%4/0_F64_0_9", "Pseudo/2_F64_2_10", "%arg9/1_None_0_12", "Pseudo/3_None_0_12", "%arg10_None_0_13", "Pseudo/4_None_0_13", "%3/1_F64_2_14", "%4/1_F64_0_15", "%5_F64_2_16", "%6_F64_0_17", "%7_F64_2_18", "%8_F64_0_19", "Pseudo/5_F64_2_20", "bedge/2_None_0_7", "bedge/4_None_0_13", "bedge/3_None_0_12", "bedge/1_None_0_6", "bedge/0_None_0_2"]
    // in_pairs: {"Arith_Addf_F64_0_4_19_0": ["%7_F64_2_18", "%6_F64_0_17"], "Affine_Load_F64_2_4_14_1": ["%arg6_F64_2_2", "%arg8_None_0_6", "%arg9/1_None_0_12"], "Affine_Forvalue_None_0_1_6_0": ["0_None_0_-1", "%2_I32Index_0_5", "1_None_0_-1"], "Arith_Mulf_F64_0_3_9_0": ["%3/0_F64_2_8", "%arg4_F64_0_2"], "Source_3": [], "Affine_Forcontrol_None_0_3_13_3": ["%arg10_None_0_13", "bedge/4_None_0_13"], "Block_Block_None_2_0_6_1": ["Pseudo/1_None_0_7", "Pseudo/3_None_0_12"], "Source_0": [], "Source_1": [], "Arith_IndexCast_I32Index_0_1_5_2": ["%arg0_I32_0_2"], "Affine_Load_F64_2_4_16_2": ["%arg7_F64_2_2", "%arg9/1_None_0_12", "%arg10_None_0_13"], "Affine_Store_F64_2_3_10_0": ["%4/0_F64_0_9", "%arg5_F64_2_2", "%arg8_None_0_6", "%arg9/0_None_0_7"], "Affine_Load_F64_2_4_18_3": ["%arg5_F64_2_2", "%arg8_None_0_6", "%arg10_None_0_13"], "Arith_IndexCast_I32Index_0_1_3_0": ["%arg1_I32_0_2"], "Arith_Mulf_F64_0_4_17_2": ["%4/1_F64_0_15", "%5_F64_2_16"], "Source_9": [], "Affine_Store_F64_2_4_20_1": ["%8_F64_0_19", "%arg5_F64_2_2", "%arg8_None_0_6", "%arg10_None_0_13"], "Affine_Forcontrol_None_0_2_7_1": ["%arg9/0_None_0_7", "bedge/2_None_0_7"], "Affine_Forcontrol_None_0_2_12_2": ["%arg9/1_None_0_12", "bedge/3_None_0_12"], "Block_Block_None_1_0_12_3": ["Pseudo/4_None_0_13"], "Arith_IndexCast_I32Index_0_1_4_1": ["%arg2_I32_0_2"], "Source_2": [], "Affine_Forvalue_None_0_2_12_2": ["0_None_0_-1", "%1_I32Index_0_4", "1_None_0_-1"], "Affine_Forvalue_None_0_3_13_3": ["0_None_0_-1", "%0_I32Index_0_3", "1_None_0_-1"], "Affine_Forcontrol_None_0_1_6_0": ["%arg8_None_0_6", "bedge/1_None_0_6"], "Block_Block_None_7_0_13_4": ["%3/1_F64_2_14", "%4/1_F64_0_15", "%5_F64_2_16", "%6_F64_0_17", "%7_F64_2_18", "%8_F64_0_19", "Pseudo/5_F64_2_20"], "Source_7": [], "Source_8": [], "Source_5": [], "Sink": [], "Source_6": [], "Affine_Load_F64_2_3_8_0": ["%arg5_F64_2_2", "%arg8_None_0_6", "%arg9/0_None_0_7"], "Source_4": [], "Block_Block_None_3_0_7_2": ["%3/0_F64_2_8", "%4/0_F64_0_9", "Pseudo/2_F64_2_10"], "Block_Block_None_4_0_2_0": ["%0_I32Index_0_3", "%1_I32Index_0_4", "%2_I32Index_0_5", "Pseudo/0_None_0_6"], "Arith_Mulf_F64_0_4_15_1": ["%arg3_F64_0_2", "%3/1_F64_2_14"], "Func_Func_None_9_0_2_0": ["%arg0_I32_0_2", "%arg1_I32_0_2", "%arg2_I32_0_2", "%arg3_F64_0_2", "%arg4_F64_0_2", "%arg5_F64_2_2", "%arg6_F64_2_2", "%arg7_F64_2_2", "bedge/0_None_0_2"], "Affine_Forvalue_None_0_2_7_1": ["0_None_0_-1", "%0_I32Index_0_3", "1_None_0_-1"]}
    // out_pairs: {"Affine_Forcontrol_None_0_1_6_0": "Pseudo/0_None_0_6", "Affine_Load_F64_2_4_18_3": "%7_F64_2_18", "Source_7": "%arg7_F64_2_2", "Affine_Store_F64_2_3_10_0": "Pseudo/2_F64_2_10", "Source_6": "%arg6_F64_2_2", "Affine_Forvalue_None_0_1_6_0": "%arg8_None_0_6", "Arith_IndexCast_I32Index_0_1_4_1": "%1_I32Index_0_4", "Affine_Forvalue_None_0_2_12_2": "%arg9/1_None_0_12", "Arith_IndexCast_I32Index_0_1_3_0": "%0_I32Index_0_3", "Source_1": "%arg1_I32_0_2", "Arith_IndexCast_I32Index_0_1_5_2": "%2_I32Index_0_5", "Arith_Mulf_F64_0_4_17_2": "%6_F64_0_17", "Source_2": "%arg2_I32_0_2", "Source_3": "%arg3_F64_0_2", "Source_8": "0_None_0_-1", "Block_Block_None_4_0_2_0": "bedge/0_None_0_2", "Block_Block_None_7_0_13_4": "bedge/4_None_0_13", "Affine_Forvalue_None_0_3_13_3": "%arg10_None_0_13", "Source_5": "%arg5_F64_2_2", "Source_4": "%arg4_F64_0_2", "Block_Block_None_1_0_12_3": "bedge/3_None_0_12", "Arith_Mulf_F64_0_3_9_0": "%4/0_F64_0_9", "Affine_Forcontrol_None_0_2_12_2": "Pseudo/3_None_0_12", "Affine_Forcontrol_None_0_3_13_3": "Pseudo/4_None_0_13", "Affine_Load_F64_2_4_16_2": "%5_F64_2_16", "Arith_Mulf_F64_0_4_15_1": "%4/1_F64_0_15", "Affine_Load_F64_2_4_14_1": "%3/1_F64_2_14", "Source_9": "1_None_0_-1", "Source_0": "%arg0_I32_0_2", "Arith_Addf_F64_0_4_19_0": "%8_F64_0_19", "Block_Block_None_3_0_7_2": "bedge/2_None_0_7", "Affine_Forvalue_None_0_2_7_1": "%arg9/0_None_0_7", "Block_Block_None_2_0_6_1": "bedge/1_None_0_6", "Affine_Forcontrol_None_0_2_7_1": "Pseudo/1_None_0_7", "Affine_Store_F64_2_4_20_1": "Pseudo/5_F64_2_20", "Affine_Load_F64_2_3_8_0": "%3/0_F64_2_8"}
    let (vertices, edges, in_pairs, out_pairs) = lib::parse_graph(&input);
    // Initialize an empty RecExpr (recursive expression) tree.
    let mut expr: RecExpr<SimpleLanguage> = RecExpr::default();
    // Create a HashMap to keep track of the mapping from graph nodes to their IDs in the expression tree.
    let mut edge_map: HashMap<String, Id> = HashMap::new();
    // Initialize a queue with the vertices, reversed to start from the end.
    let mut queue: Vec<String> = vertices.clone();
    queue.reverse();

    while !queue.is_empty(){
        let operation=queue.pop().unwrap();    
        if let Some(input_edges) = in_pairs.get(&operation) {
            //Handling the operation without input, like Source_# operation
            if input_edges.is_empty() {
                if let Some(des) = out_pairs.get(&operation) {
                    let parts: Vec<&str> = des.split('_').collect();
                    let new_des: String = if let Some(first_part) = parts.get(0) {
                        if first_part.parse::<f64>().is_ok() || first_part.parse::<i32>().is_ok() || first_part.contains('%') {
                            first_part.to_string()
                        } else {
                            des.clone()
                        }
                    } else {
                        des.clone()
                    };
                    if (new_des.contains('+') || new_des.contains('-') || new_des.contains('*') || new_des.contains('/')) 
                    && !new_des.parse::<i32>().is_ok() 
                    && !new_des.parse::<i64>().is_ok()
                    && !new_des.parse::<f32>().is_ok()
                    && !new_des.parse::<f64>().is_ok() {
                        // true branch
                        // new_des contains one of the specified characters and cannot be parsed as i32 or i64
                    } else {
                        let temp_id = expr.add(SimpleLanguage::Symbol(new_des.clone().into()));
                        edge_map.insert(des.clone(), temp_id);
                        // false branch
                        // either new_des does not contain the specified characters, or it can be parsed as i32 or i64
                    }
                }
                queue.retain(|x| x != &operation);
            }
            // Handleing normal operation
            else{
                let mut in_id = Vec::new();
                for e in input_edges{
                    // println!("{:?}",e);
                    if let Some(id) = edge_map.get(e) {
                        in_id.push(*id);
                    }
                    else{
                        let op = e.split('_').next().unwrap_or_else(|| panic!("Cannot find id for {}", e));
                        let convert_op = lib::convert_expression(op);
                        if convert_op.is_empty() || convert_op == "some_error_indicator" {
                            panic!("Cannot find id for {}", e);
                        }
                        // Proceed with using convert_op
                        println!("Converted operation: {}", convert_op);
                        let parsed_expr: RecExpr<SimpleLanguage> = convert_op.parse().unwrap();
                        let mut id_map: HashMap<Id, Id> = HashMap::new();
                        for (idx, enode) in parsed_expr.as_ref().iter().enumerate() {
                            let mapped_children = enode.children().iter()
                                .map(|&child| *id_map.get(&child).expect("Child not found in id_map"))
                                .collect::<Vec<_>>(); 
                            // println!("idx: {:?}, enode: {:?}, mapped_children: {:?},id_map: {:?}",idx, enode, mapped_children, id_map);
                            // let new_id = expr.add(enode.clone().map_children(|i| mapped_children[usize::from(i)]));
                            let new_id = expr.add(enode.clone().map_children(|i| {
                                *id_map.get(&i).unwrap_or_else(|| panic!("Index out of bounds or child not found for Id {:?}. Mapped: {:?}", i, id_map))
                            }));
                            id_map.insert(Id::from(idx), new_id);
                        }
                        let last_index = expr.as_ref().len() - 1;
                        let last_id = Id::from(last_index);                        
                        edge_map.insert(e.clone(), last_id);
                        in_id.push(last_id);
                        // panic!("C123124123 {}", e);
                    }
                }
                let name: Vec<&str>;
                if operation.starts_with("Block") || operation.starts_with("Func") {
                    name = operation.rsplitn(5, '_').collect();
                } else {
                    name = operation.rsplitn(4, '_').collect();
                }
                let stripped_name = name.last().unwrap().to_lowercase();
                if let Some(value) = convert_to_simple_language_enum (in_id, &stripped_name) {
                    // println!("stripped_name:{:?}",stripped_name);
                    let temp_id = expr.add(value);
                    if let Some(des)=out_pairs.get(&operation){
                        edge_map.insert(des.clone(), temp_id);
                    }
                    queue.retain(|x| x != &operation);
                }
                else {
                    panic!("Unknown enum variant for {}", stripped_name);
                }
            }
        } else {
            println!("Key not found");
        }
    }
    expr
}


struct CustomCost;

impl CostFunction<SimpleLanguage> for CustomCost {
    type Cost = isize;

    fn cost<C>(&mut self, enode: &SimpleLanguage, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost
    {
        let ast_size = -1 + enode.children().iter().map(|&id| costs(id) as isize).sum::<isize>();
        (ast_size) as Self::Cost
    }
}

// struct SillyCostFn;
// impl CostFunction<SimpleLanguage> for SillyCostFn {
//     type Cost = f64;
//     fn cost<C>(&mut self, enode: &SimpleLanguage, mut costs: C) -> Self::Cost
//     where
//         C: FnMut(Id) -> Self::Cost
//     {
//         let op_cost = match enode.as_str() {
//             "foo" => 100.0,
//             "bar" => 0.7,
//             _ => 1.0
//         };
//         enode.fold(op_cost, |sum, id| sum + costs(id))
//     }
// }

// impl<'a> CustomCost<'a> {
//     // A helper function to calculate the AST size.
//     fn size<C>(&self, enode: &SimpleLanguage, mut costs: C) -> usize
//     where
//         C: FnMut(Id) -> usize,
//     {
//         1 + enode.children().iter().map(|&id| costs(id)).sum::<usize>()
//     }

//     fn cost<C>(&mut self, enode: &SimpleLanguage, mut costs: C) -> f64
//     where
//         C: FnMut(Id) -> usize,
//     {
//         let size = self.size(enode, &mut costs);

//         // Invert and scale the size to fit into a usize.
//         // This is an example scaling, and you might need to adjust it.
//         if size > 0 {
//             1.0 / size
//         } else {
//             usize::MAX // Handle division by zero if needed
//         }
//     }
// }


fn simplify_programmatically(expr: &RecExpr<SimpleLanguage>) -> (isize, RecExpr<SimpleLanguage>,Runner<SimpleLanguage,()>){
    // Simplify the expression using a Runner
    let runner = Runner::default().with_expr(expr).with_iter_limit(1).run(&make_rules());


    // let egraph = Runner::default().with_egraph(egraph).run(rules).egraph;
    // The Runner knows which e-class the expression given with `with_expr` is in
    let root = runner.roots[0];

    // Use an Extractor to pick the best element of the root eclass
    let egraph_clone = runner.egraph.clone();
    // let cost_func = CustomCost { egraph: &egraph_clone };
    let extractor = Extractor::new(&runner.egraph, CustomCost);
    let (best_cost, best) = extractor.find_best(root);

    (best_cost, best,runner)
}

#[test]
fn simple_tests() {
    let expr = build_expr();
    let (best_cost, best,runner) = simplify_programmatically(&expr);
    println!("{}",expr);
    println!("{}",best);
    println!("{}",best_cost);
    println!("{}",CustomCost.cost_rec(&expr));

    
    // let egraph= runner.egraph;
    // println!("{:?}",&runner);
    // for eclass in egraph.classes() {
    //     let eclass_id = eclass.id; // Get the EClassId directly from the eclass
    //     println!("E-class ID: {:?}", eclass_id);
    
    //     // Iterate over each e-node in the e-class
    //     for enode in eclass.iter() {
    //         println!("  E-node: {:?}", enode);
    //     }
    // }

    // egraph.dot().to_dot("target/foo5.dot");
}


