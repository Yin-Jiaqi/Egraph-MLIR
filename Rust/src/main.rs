use Mlir_Egg::parse_graph;
use regex::Regex;
use std::collections::HashMap;
use std::fs;
use egg::*;



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
        

        
        
        if let Some(value) = language_enum {
            let temp_id = expr.add(value);
            node_map.insert(src, temp_id);
        } else {
            println!("Unknown enum variant.");
        }
    }
}