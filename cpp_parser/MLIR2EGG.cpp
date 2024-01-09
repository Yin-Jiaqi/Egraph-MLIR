#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <regex>
#include <functional>
#include <cerrno>
#include <cassert>

#include "include/utilities.hpp"
#include "include/graph.hpp"
#include "include/parser.hpp"
// 140


const std::string filepath = "mlir/gemm/gemm.mlir"; 

/**
 * In this case, checks if the string is empty or contains only whitespace.
 *
 * @param str The string to be checked against the pattern.
 * @return true if the string matches the pattern, false otherwise.
 */
bool checkPattern(const std::string& str) {
    static const std::regex emptyLinePattern(R"(^\s*$)");
    return std::regex_match(str, emptyLinePattern);
}




int main() {
    // Initialize graph and vertices
    Graph empty_graph;
    Boost_Graph g(empty_graph);
    std::vector<Edge*> edges_to_delete;
    std::vector<Vertex*> vertices_to_delete;

    // Create and add unknown sink vertex, all vertex/operation that has no output data will connect to unknown sink
    Vertex unknown_sink("Sink", "SS", "Sink", "None", {}, {}, 0, -1, 0);
    g.addOrUpdateVertex(&unknown_sink);
    std::vector<Vertex*> sink_op = {&unknown_sink};

    // Create and add a new source vertex
    // An operation always produces at most one output, though it can have several inputs. Thus, we establish a universal Sink node linked to every vertex or operation lacking output data. However, for data that we cannot explicitly find the operation that produce it, we will allocate a seperate Source node.
    // auto vector_size = g.getVertexList().at("SS").at("Source").size();
    // std::unique_ptr<Vertex> source_for_block = std::make_unique<Vertex>("Source_0", "SS", "Source", "None", std::vector<Edge*>{}, std::vector<Edge*>{}, 0, -1, 0);
    // g.addOrUpdateVertex(source_for_block.get());
    // std::unique_ptr<Edge> source_edge = std::make_unique<Edge>();
    // source_edge->set("No_block", source_for_block.get(), sink_op, "Block", 0, -1);
    // g.addEdge2Graph(source_edge.get());
    // source_for_block->updateOutput(source_edge.get());


    // Initialize variables for processing the file
    std::string line;
    int line_num = 1;
    std::vector<Block*> todo_blist = {};
    std::vector<Block*> done_blist = {};

    // File processing logic
    std::ifstream inFile = utilities::openFile(filepath);
    while (std::getline(inFile, line)) {
        if (!line.empty()) {
            // Skip the line of "return" and the attributes which identify the hardware information
            // module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
            if (line.find("module attributes") != std::string::npos || line.find("return") != std::string::npos){
            }
            // Regex R"(^\s*\}\s*$)" matches a line with only whitespace and a single '}' character.
            // "}" ends code blocks, e.g., functions, loops.
            else if (std::regex_match(line, std::regex(R"(^\s*\}\s*$)"))) {
                if (inFile.peek() != '\n' && inFile.peek() != EOF) {
                    auto block= todo_blist.back();
                    block->setPosend(line_num);
                    todo_blist.pop_back();
                    done_blist.push_back(block);
                }
            }
            // Function and input data defination
            // func.func @kernel_gemm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f64, %arg4: f64, %arg5: memref<?x1100xf64>, %arg6: memref<?x1200xf64>, %arg7: memref<?x1100xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
            else if (line.find("func.func")!= std::string::npos){
                MyTupleType2 tuple_output = Func::split_data(line);
                Func func;
                Vertex* func_op = func.parse(line,tuple_output,g,line_num,todo_blist);
                // func.func should contain "{" symbol to generate a block
                if (line.find("{")!= std::string::npos){
                    int index=todo_blist.size()+done_blist.size();
                    Block* block = new Block("block"+std::to_string(index), index, line_num, -1, "", "", "", func_op);
                    todo_blist.push_back(block);
                }
                else{
                    throw std::runtime_error(std::string("Input format error: function should contain a Block") + line);
                }
            }
            // parsing for scf.for and affine.for
            else if (line.find(".for")!= std::string::npos){
                MyTupleType3 tuple_output = For::split_data(line);
                For ForInstance;
                Vertex* for_op_control = ForInstance.parse_for(line,tuple_output,g,line_num,todo_blist,"value");
                // scf.for and affine.for should contain "{" symbol to generate a block
                if (line.find("{")!= std::string::npos){
                    int index=todo_blist.size()+done_blist.size();
                    std::string loop_start=utilities::last_n_from_tuple<1>(tuple_output).at(0);
                    std::string loop_end=utilities::last_n_from_tuple<1>(tuple_output).at(1);
                    std::string loop_step=utilities::last_n_from_tuple<1>(tuple_output).at(2);
                    Block* block = new Block("block"+std::to_string(index)+"<"+loop_start+"><"+loop_end+"><"+loop_step+">", index, line_num, -1, loop_start, loop_end, loop_step, for_op_control);
                    todo_blist.push_back(block);
                }
                else{
                    throw std::runtime_error(std::string("Input format error: for loop should contain a Block") + line);
                }
            }
            // parsing for computation operation in arith, affine, etc. Like arith.muli, arith.addi, affine.store.
            else if (!std::regex_match(line, std::regex(R"(^\s*$)"))) {
                MyTupleType1 tuple_output = Dialect::analyzeMLIR(line, Dialect::splitPrimary, Dialect::analyzeOperation, Dialect::analyzeType);
                Computation computation;
                computation.parse(line,tuple_output,g,line_num,todo_blist);
            }
        }
        line_num+=1;
    }

    for (Block* block : todo_blist) {
        delete block;
    }
    todo_blist.clear();

    // Handling the processing of Block
    //   ----------> block_op ----------> knob
    //   operations           des1(bedge)
    for (auto block: done_blist){
        auto operations = block->getInEdge();
        auto line_num_block = block->getPos()[0];
        Edge* des1 = new Edge;
        // For Block operation, the dimension is set the number of input operation (operations.size())
        Vertex* block_op = new Vertex(std::string("Block_Block_") + std::to_string(block->getIndex()), 
                        "Block", "Block", "None",operations,
                        std::vector<Edge*>{des1}, operations.size(), line_num_block, todo_blist.size());
        auto knob=block->getKnob();
        std::vector<Vertex*> vertices{knob};
        knob->updateInput(des1);
        std::string ename = "bedge/"+std::to_string(block->getIndex());
        des1->set(ename, block_op, vertices, "None",0, line_num_block);
        g.addEdge2Graph(des1);
        g.addOrUpdateVertex(block_op);
        for (size_t i = 0; i < operations.size(); ++i) {
            operations[i]->updateOutOp(block_op,&(g.getGraph()));
        }
        edges_to_delete.push_back(des1);
        vertices_to_delete.push_back(block_op);
    }



    // Print the output data.
    std::ofstream file("output_test.txt");
    if (file.is_open()) {
        g.printGraph(file);
        file.close();
    } else {
        std::cerr << "Unable to open file" << std::endl;
    }
    for (Block* block : done_blist) {
        delete block;
    }
    done_blist.clear();
    for (Edge* edge : edges_to_delete) {
        delete edge;
    }
    edges_to_delete.clear();
    for (Vertex* vertex : vertices_to_delete) {
        delete vertex;
    }
    vertices_to_delete.clear();
    done_blist.clear();
    inFile.close();
      return 0;
}