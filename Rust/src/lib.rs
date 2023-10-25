use regex::Regex;
use std::collections::HashMap;
use std::fs;
use library::match_simple_language_enum;


pub fn parse_graph(input: &str) -> (Vec<String>, Vec<String>, HashMap<String, Vec<String>>, Vec<(String, String)>) {
    let mut vertices = Vec::new();
    let mut edges = Vec::new();
    let mut in_pairs: HashMap<String, Vec<String>> = HashMap::new();
    let mut out_pairs = Vec::new();

    // Patterns
    let vertex_pattern = Regex::new(r"Name: (?P<name>.*?_.*?)\s+Dtype: (?P<dtype>.*)\s+Dimension: (?P<dim>.*)\s+#Line: (?P<line>.*)\s+Input Edges: (?P<input_edges>.*?)\s+Output Edges: (?P<output_edges>.*?)\s*(?:\n|$)").unwrap();

    let edge_pattern = Regex::new(r"Edge Name: (?P<name>.*)\s+Dtype: (?P<dtype>.*)\s+Dimension: (?P<dim>.*)\s+Source Vertex: (?P<source_vertex>.*)\s+#Line: (?P<line>.*)\s+Target Vertices: (?P<target_vertices>.*)").unwrap();


    let mut vertex_map = HashMap::new();
    let mut edge_map = HashMap::new();

    for cap in vertex_pattern.captures_iter(input) {
        let name = &cap["name"];
        let dtype = &cap["dtype"];
        let dim = &cap["dim"];
        let line = &cap["line"];

        let parts: Vec<&str> = name.split('_').collect();
        let operation = parts[0..parts.len() - 1].join("_");
        let index = parts.last().unwrap();

        // Formatting vertices
        let formatted_vertex = if name == "Source" || name == "Sink" {
            name.to_string()
        } else {
            format!("{}_{}_{}_{}_{}", operation, dtype, dim, line, index)
        };
        vertex_map.insert(name.to_string(), formatted_vertex.clone());
        vertices.push(formatted_vertex);
    }

    for cap in edge_pattern.captures_iter(input) {
        let name = &cap["name"];
        let dtype = &cap["dtype"];
        let dim = &cap["dim"];
        let line  = &cap["line"];

        // Formatting edges
        let formatted_edge = format!("{}_{}_{}_{}", name, dtype, line, dim);
        edge_map.insert(name.to_string(), formatted_edge.clone());
        edges.push(formatted_edge);
    }

    // for key in vertex_map.keys() {
    //     println!("1234 {}", key);
    // }

    // Creating pairs
    let pairing_pattern = Regex::new(r"Name: (?P<vname>.*?_.*?)\s+Dtype: (?P<dtype>.*?)\s+Dimension: (?P<dim>.*?)\s+#Line: (?P<line>.*?)\s+Input Edges: (?P<iedges>.*?)\s+Output Edges: (?P<oedges>.*?)\s*(?:\n|$)").unwrap();


    for cap in pairing_pattern.captures_iter(input) {
        let vname = &cap["vname"];
        let iedges: Vec<&str> = cap["iedges"].split_whitespace().collect();
        let oedges: Vec<&str> = cap["oedges"].split_whitespace().collect();
        let vertex_key = vertex_map[vname].clone();
        let mut incoming_edges = Vec::new();
        for &edge in &iedges {
            incoming_edges.push(edge_map[edge].clone());
        }
        // for key in edge_map.keys() {
        //     println!("{}", key);
        // }
        in_pairs.insert(vertex_key, incoming_edges);

        for &edge in &oedges {
            out_pairs.push((vertex_map[vname].clone(), edge_map[edge].clone()));
        }        
    }


    (vertices, edges, in_pairs, out_pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let input = fs::read_to_string("output111.txt").expect("Failed to read file");
        let (vertices, edges, in_pairs, out_pairs) = parse_graph(&input);
        println!("\nIngoing Pairs!!");
        for (vertex, incoming_edges) in &in_pairs {
            println!("Vertex: {}", vertex);
            for edge in incoming_edges {
                println!("    Incoming Edge: {}", edge);
            }
        }

        println!("\nVertices!");
        for v in &vertices {
            println!("({})", v);
        }

        println!("\nEdges!");
        for e in &edges {
            println!("({})", e);
        }

        println!("\nOutgoing Pairs!");
        for (src, dst) in &out_pairs {
            println!("({}, {})", src, dst);
        }
    }
}


#[test]
fn testt() {
    let vertex_pattern = Regex::new(r"Name: (?P<name>\w+_\d+)\s+Dtype: (?P<dtype>\w+)\s+Dimension: (?P<dim>-?\d+)\s+#Line: (?P<line>\d+)\s+Input Edges: (?P<input_edges>[%\w\s]+)\s+Output Edges: (?P<output_edges>[%\w\s]+)").unwrap();

    let text = r"Name: Arith_Constant_0
Dtype: Index
Dimension: 0
#Line: 2
Input Edges: 0 block_0 
Output Edges: %c0 ";

    if vertex_pattern.is_match(&text) {
        println!("The text matches the pattern.");
    } else {
        println!("The text does NOT match the pattern.");
    }
}

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
