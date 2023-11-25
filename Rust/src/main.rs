mod lib;
use regex::Regex;
use std::collections::HashMap;
use std::fs;
use egg::*;
use Mlir_Egg::{SimpleLanguage, convert_to_simple_language_enum}; // Add other necessary imports

fn make_rules() -> Vec<Rewrite<SimpleLanguage, ()>> {
    vec![
        // rewrite!("block-expand"; "(block_block_none ?a ?b)" => "(block_block_none ?a)"),

        rewrite!("rolling_0"; "(affine_forcontrol_none_0 (affine_forvalue_none_0 ?op1 ?op2 ?op3) (block_block_none ?op4))" => "(affine_forcontrol_none_0 (affine_forvalue_none_0 ?op1 ?op2 (* 2 ?op3)) (block_block_none ?op4 ?op4))"),
        
        // rewrite!("rolling_0"; "(affine_forcontrol_none_0 (affine_forvalue_none_0 ?op1 ?op2 ?op3) (block_block_none ?op4))" => "(affine_forcontrol_none_0 (affine_forvalue_none_0 ?op1 ?op2 64) (block_block_none (affine_forcontrol_none_0 (affine_forvalue_none_0 (affine_forvalue_none_0 ?op1 ?op2 64) (+ (affine_forvalue_none_0 ?op1 ?op2 64) 64) 1) (block_block_none ?op4))))"), 
    ]
}


fn build_expr() -> RecExpr<SimpleLanguage> {
    let input = fs::read_to_string("output.txt").expect("Failed to read file");
    let (vertices, edges, in_pairs, out_pairs) = lib::parse_graph(&input);
    let mut expr: RecExpr<SimpleLanguage> = RecExpr::default();
    let mut edge_map: HashMap<String, Id> = HashMap::new();
    let mut queue: Vec<String> = vertices.clone();
    queue.reverse();

    while !queue.is_empty(){
        let operation=queue.pop().unwrap();
        // println!("operation{}", operation);
        if let Some(input_edges) = in_pairs.get(&operation) {
            // if input_edges.is_empty(){
            //     if let Some(des)=out_pairs.get(&operation){
            //         let parts: Vec<&str> = des.split('_').collect();
            //         if let Some(first_part) = parts.get(0) {
            //             if first_part.parse::<i32>().is_ok() || first_part.contains('%') {
            //                 des = first_part.to_string();
            //             }
            //         }
            //         println!("{}",des);
            //         let temp_id = expr.add(SimpleLanguage::Symbol(des.clone().into()));
            //         edge_map.insert(des.clone(), temp_id);
            //     }
            //     queue.retain(|x| x != &operation);
            // }
            if input_edges.is_empty() {
                if let Some(des) = out_pairs.get(&operation) {
                    let parts: Vec<&str> = des.split('_').collect();
                    let new_des = if let Some(first_part) = parts.get(0) {
                        if first_part.parse::<i32>().is_ok() || first_part.contains('%') {
                            first_part.to_string()
                        } else {
                            des.clone()
                        }
                    } else {
                        des.clone()
                    };
            
                    println!("{}", new_des);
                    let temp_id = expr.add(SimpleLanguage::Symbol(new_des.clone().into()));
                    edge_map.insert(des.clone(), temp_id);
                }
                queue.retain(|x| x != &operation);
            }
            else{
                let mut in_id = Vec::new();
                for e in input_edges{
                    if let Some(id) = edge_map.get(e) {
                        in_id.push(*id);
                    }
                    else{
                        panic!("Cannot find id for {}", e)
                    }
                }
                let name: Vec<&str>;
                if operation.starts_with("Block") || operation.starts_with("Func") {
                    name = operation.rsplitn(5, '_').collect();
                } else {
                    name = operation.rsplitn(4, '_').collect();
                }
                
                // println!("name is {:?}", name);
                // println!("operation is start with block: {}", operation.starts_with("Block"));
                let stripped_name = name.last().unwrap().to_lowercase();
                // println!("stripped_name is {}", stripped_name);
                if let Some(value) = convert_to_simple_language_enum (in_id, &stripped_name) {
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

fn simplify_programmatically(expr: &RecExpr<SimpleLanguage>) -> (RecExpr<SimpleLanguage>,Runner<SimpleLanguage,()>){
    // Simplify the expression using a Runner
    let runner = Runner::default().with_expr(expr).run(&make_rules());


    // let egraph = Runner::default().with_egraph(egraph).run(rules).egraph;
    // The Runner knows which e-class the expression given with `with_expr` is in
    let root = runner.roots[0];

    // Use an Extractor to pick the best element of the root eclass
    let extractor = Extractor::new(&runner.egraph, AstDepth);
    let (best_cost, best) = extractor.find_best(root);

    (best,runner)
}

#[test]
fn simple_tests() {
    let expr = build_expr();
    let (best,runner) = simplify_programmatically(&expr);
    println!("{}",expr);
    println!("{}",best);
    let egraph= runner.egraph;
    for eclass in egraph.classes() {
        let eclass_id = eclass.id; // Get the EClassId directly from the eclass
        println!("E-class ID: {:?}", eclass_id);
    
        // Iterate over each e-node in the e-class
        for enode in eclass.iter() {
            println!("  E-node: {:?}", enode);
        }
    }
    // egraph.dot().to_png("target/f123456.png");
}