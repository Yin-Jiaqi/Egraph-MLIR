use regex::Regex;
// use std::collections::HashMap;
use std::fs;
use std::collections::{HashMap, VecDeque};

fn topological_sort(ingoing: &mut HashMap<String, Vec<String>>,outgoing: &HashMap<String,String>) -> Vec<String> {
    let mut in_degree: HashMap<String, usize> = HashMap::new();
    let mut queue: VecDeque<String> = VecDeque::new();
    let mut result: Vec<String> = Vec::new();
    let mut marked_edge:Vec<String> = Vec::new();

    // Initialize in_degree
    for vertex in ingoing.keys() {
        in_degree.insert(vertex.clone(), ingoing[vertex].len());
        if ingoing[vertex].len() == 0 {
            queue.push_back(vertex.clone());
        }
    }


    while let Some(vertex) = queue.pop_front() {
        result.push(vertex.clone());
        ingoing.remove(&vertex);
        if let Some(edge) = outgoing.get(&vertex) {
            marked_edge.push(edge.clone());
        }
    }

    // Iterate over vertices which have an edge from the current vertex
    // for (des, edges) in ingoing.iter() {
        while !ingoing.is_empty() {
            let mut dess: Option<String> = None; // Change the type to Option<String>
            
            for (des, edges) in ingoing.iter() {
                let mut flag = true;
                
                for edge in edges.iter() {
                    if !marked_edge.contains(edge) {
                        flag = false;
                        break; // break out of the inner loop when the flag is set to false
                    }
                }
                
                if flag {
                    dess = Some(des.to_string());
                    break;
                }
            }
        
            let dess = match dess {
                Some(val) => val,
                None => panic!("No suitable value found for dess!"),
            };
            
            result.push(dess.clone());
            ingoing.remove(&dess);
            
            if let Some(edge) = outgoing.get(&dess) {
                if marked_edge.contains(edge) {
                    panic!("Error: There is a cycle {}", edge);
                } else {
                    marked_edge.push(edge.clone());
                }
            }
        }        
    result
}


pub fn parse_graph(input: &str) -> (Vec<String>, Vec<String>, HashMap<String, Vec<String>>, HashMap<String,String>) {
    let mut vertices = Vec::new();
    let mut edges = Vec::new();
    let mut in_pairs: HashMap<String, Vec<String>> = HashMap::new();
    let mut out_pairs: HashMap<String,String> = HashMap::new();

    // Patterns

    // let vertex_pattern = Regex::new(r"Name: (?P<name>.*?_.*?)\s+Dtype: (?P<dtype>.*)\s+Dimension: (?P<dim>.*)\s+#Line: (?P<line>.*)\s+#Block: (?P<blk>.*)\s+Input Edges: (?P<iedges>.*?)\s+Output Edges: (?P<oedges>.*?)\s*(?:\n|$)").unwrap();

    let vertex_pattern = Regex::new(r"Name: (?P<name>.*?)\s+Dtype: (?P<dtype>.*)\s+Dimension: (?P<dim>.*)\s+#Line: (?P<line>.*)\s+#Block: (?P<blk>.*)\s+Input Edges: (?P<iedges>.*?)\s+Output Edges: (?P<oedges>.*?)\s*(?:\n|$)").unwrap();

    let edge_pattern = Regex::new(r"Edge Name: (?P<name>.*)\s+Dtype: (?P<dtype>.*)\s+Dimension: (?P<dim>.*)\s+Source Vertex: (?P<source_vertex>.*)\s+#Line: (?P<line>.*)\s+Target Vertices: (?P<target_vertices>.*)").unwrap();


    let mut vertex_map = HashMap::new();
    let mut edge_map = HashMap::new();



    for cap in edge_pattern.captures_iter(input) {
        let name = &cap["name"];
        let dtype = &cap["dtype"];
        let dim = &cap["dim"];
        let line  = &cap["line"];

        // Formatting edges
        let formatted_edge = format!("{}_{}_{}_{}", name, dtype, dim, line);
        edge_map.insert(name.to_string(), formatted_edge.clone());
        edges.push(formatted_edge);
    }

    for cap in vertex_pattern.captures_iter(input) {
        let name = &cap["name"];
        if true{
            let dtype = &cap["dtype"];
            let line = &cap["line"];
            let blk= &cap["blk"];    
            let parts: Vec<&str> = name.split('_').collect();
            let operation = parts[0..parts.len() - 1].join("_");
            let index = parts.last().unwrap();
            let dim: &str;
            let len_string=cap["iedges"].split_whitespace().collect::<Vec<&str>>().len().to_string();
            if operation=="Func_Func"{
                dim = &len_string;
            }
            else{
                dim = &cap["dim"];
            }
    
            // Formatting vertices
            // let formatted_vertex = if name == "Source" || name == "Sink" {
            let formatted_vertex = if name == "Sink" || name.contains("Source") {
                name.to_string()
            } else {
                format!("{}_{}_{}_{}_{}_{}", operation, dtype, dim, blk, line, index)
            };
            vertex_map.insert(name.to_string(), formatted_vertex.clone());
            vertices.push(formatted_vertex);
    
            let iedges: Vec<&str> = cap["iedges"].split_whitespace().collect();
            let oedges: Vec<&str> = cap["oedges"].split_whitespace().collect();
        }
    }

    // println!("\nVertices!");
    // for v in &vertices {
    //     println!("{}", v);
    // }

    // println!("\nEdges!");
    // for e in &edges {
    //     println!("{}", e);
    // }

    // Creating pairs
    let pairing_pattern = Regex::new(r"Name: (?P<name>.*?)\s+Dtype: (?P<dtype>.*)\s+Dimension: (?P<dim>.*)\s+#Line: (?P<line>.*)\s+#Block: (?P<blk>.*)\s+Input Edges: (?P<iedges>.*?)\s+Output Edges: (?P<oedges>.*?)\s*(?:\n|$)").unwrap();


    for cap in pairing_pattern.captures_iter(input) {
        let vname = &cap["name"];
        if true{
            let iedges: Vec<&str> = cap["iedges"].split_whitespace().collect();
            let oedges: Vec<&str> = cap["oedges"].split_whitespace().collect();
            let vertex_key = vertex_map[vname].clone();
            let mut incoming_edges = Vec::new();
            for &edge in &iedges {
                match edge_map.get(edge) {
                    Some(value) => incoming_edges.push(value.clone()),
                    None => {
                        // eprintln!("Error: Cannot find key {}", edge);
                        // Optionally, if you want to stop execution when an error occurs:
                        panic!("Error: Cannot find key {}", edge);
                    }
                }
            }            
            // for key in edge_map.keys() {
            //     println!("{}", key);
            // }
            in_pairs.insert(vertex_key, incoming_edges);

            for &edge in &oedges {
                out_pairs.insert(vertex_map[vname].clone(),edge_map[edge].clone());
            }
        }
    }

    let sort=topological_sort(&mut in_pairs.clone(), &out_pairs.clone());

    (sort, edges, in_pairs, out_pairs)
}



// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn it_works() {
//         let input = fs::read_to_string("output111.txt").expect("Failed to read file");
//         let (vertices, edges, in_pairs, out_pairs) = parse_graph(&input);
//         println!("\nIngoing Pairs!!");
//         for (vertex, incoming_edges) in &in_pairs {
//             println!("Vertex: {}", vertex);
//             // for edge in incoming_edges {
//             //     println!("    Incoming Edge: {}", edge);
//             // }
//             println!("{:?}", incoming_edges);
//         }

//         println!("\nSorted Vertices!");
//         println!("{:?}", vertices);
//         // for v in &vertices {
//         //     println!("{:?}", v);
//         // }

//         println!("\nEdges!");
//         println!("{:?}", edges);
//         // for e in &edges {
//         //     println!("{:?}", e);
//         // }

//         println!("\nOutgoing Pairs!");
//         println!("{:?}", out_pairs);
//         // for (src, dst) in &out_pairs {
//         //     println!("{}, {}", src, dst);
//         // }

//     }
// }


// #[test]
// fn testt() {
//     let vertex_pattern = Regex::new(r"Name: (?P<name>\w+_\d+)\s+Dtype: (?P<dtype>\w+)\s+Dimension: (?P<dim>-?\d+)\s+#Line: (?P<line>\d+)\s+Input Edges: (?P<input_edges>[%\w\s]+)\s+Output Edges: (?P<output_edges>[%\w\s]+)").unwrap();

//     let text = r"Name: Arith_Constant_0
// Dtype: Index
// Dimension: 0
// #Line: 2
// Input Edges: 0 block_0 
// Output Edges: %c0 ";

//     if vertex_pattern.is_match(&text) {
//         println!("The text matches the pattern.");
//     } else {
//         println!("The text does NOT match the pattern.");
//     }
// }

// pub fn add(left: usize, right: usize) -> usize {
//     left + right
// }



// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn it_works() {
//         let result = add(2, 2);
//         assert_eq!(result, 4);
//     }
// }
