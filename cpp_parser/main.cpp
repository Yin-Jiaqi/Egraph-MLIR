#include <iostream>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cassert>
#include <vector>
#include <tuple>
#include <functional>
#include "/home/jiaqiyin/MLIR_project/IR2expr5/include/parser.hpp"
#include "/home/jiaqiyin/MLIR_project/IR2expr5/include/utilities.hpp"
#include "/home/jiaqiyin/MLIR_project/IR2expr5/include/graph.hpp"
#include <regex>

const std::string filepath = "mlir/gemm/gemm.mlir"; 

bool checkPattern(const std::string& str) {
    std::regex pattern(R"(^\s*$)");
    return std::regex_match(str, pattern);
}

int main() {
    Graph empty_graph; Boost_Graph g(empty_graph);
    Vertex unknown_sink ("Sink","SS","Sink","None",{}, {},0, -1, 0);
    g.add_or_update_vertex(&unknown_sink);// specify your file path here
    Edge* source_edge = new Edge;
    std::vector<Vertex*> sink_op = std::vector<Vertex*> {&unknown_sink};
    auto vector_size = g.getVertexList().find("SS")->second.find("Source")->second.size();
    Vertex* source_for_block = new Vertex("Source_" + std::to_string(vector_size),"SS","Source","None",{}, {},0, -1, 0);
    g.add_or_update_vertex(source_for_block);
    source_edge->set("No_block", source_for_block, sink_op, "Block",0,-1);
    g.add_edge_to_graph(source_edge);
    source_for_block->update_output(source_edge);
    int file_length=utilities::countLines(filepath);
    std::ifstream inFile(filepath);
    if (!inFile) {
        std::cerr << "Unable to open file: " << filepath << std::endl;
        std::cerr << "Reason: " << strerror(errno) << std::endl;
        return 1;  // Exit with an error code
    }

    std::string line;
    int line_num=1;
    std::vector<Block*> todo_blist={};
    std::vector<Block*> done_blist={};
    while (std::getline(inFile, line)) {
        std::cout<<line<<std::endl;
        if (!line.empty()) {
            if (line.find("module attributes")!= std::string::npos){
            }
            else if (line.find("return")!= std::string::npos){ }
            else if (std::regex_match(line, std::regex(R"(^\s*\}\s*$)"))) {
                if (inFile.peek() != '\n' && inFile.peek() != EOF) {
                    auto block= todo_blist.back();
                    block->setPosend(line_num);
                    todo_blist.pop_back();
                    done_blist.push_back(block);
                }
            }
            else if (line.find("func.func")!= std::string::npos){
                MyTupleType2 tuple_output = Func::split_data(line);
                Func func;
                Vertex* func_op = func.parse(line,tuple_output,g,line_num,todo_blist);
                if (line.find("{")!= std::string::npos){
                    int index=todo_blist.size()+done_blist.size();
                    Block* block = new Block("block"+std::to_string(index), index, line_num, -1, "", "", "", func_op);
                    todo_blist.push_back(block);
                    //Block(const std::string& bname, int idx, int x, int y, int iter1, int iter2, int iter3, Vertex* knob = nullptr) 
                }
                else{
                    throw std::runtime_error(std::string("Input format error: function should contain a Block") + line);
                }
            }
            else if (line.find(".for")!= std::string::npos){
                MyTupleType3 tuple_output = For::split_data(line);
                For ForInstance;
                Vertex* for_op_control = ForInstance.parse_for(line,tuple_output,g,line_num,todo_blist,"value");
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
            else if (!std::regex_match(line, std::regex(R"(^\s*$)"))) {
                MyTupleType1 tuple_output = Dialect::analyzeMLIR(line, Dialect::splitPrimary, Dialect::analyzeOperation, Dialect::analyzeType);
                // std::cout<<std::get<0>(tuple_output)<<std::endl;
                Computation computation;
                computation.parse(line,tuple_output,g,line_num,todo_blist);
            }
        }
        line_num+=1;
    }


    for (auto block: done_blist){
        auto operations = block->getOperations();
        Edge* des1 = new Edge;
        auto line_num_block = block->getPos()[0];
        // auto vector_size = g.getVertexList().find("Block")->second.find("Block")->second.size();
        Vertex* block_op = new Vertex(std::string("Block_Block_") + std::to_string(block->getIndex()), 
                        "Block", "Block", "None",operations,
                        std::vector<Edge*>{des1}, operations.size(), line_num_block, todo_blist.size());
        auto knob=block->getKnob();
        std::vector<Vertex*> vertices{knob}; // Create a named vector with 'knob' as its initial element
        knob->update_input(des1);
        std::string ename = "bedge/"+std::to_string(block->getIndex());
        // if (g.edge_exists_l2(ename)){
        //     auto name2Edge = g.getname2Edge();
        //     auto e = name2Edge.at(ename).at(ename);
        //     e->setEdgeName(ename+"/0");
        //     name2Edge[ename].erase(ename);
        //     name2Edge[ename][ename+"/0"] = e;
        //     ename=ename+"/1";
        // }
        // else if (g.edge_exists_l2(ename+"/0")){
        //     auto name2Edge = g.getname2Edge();
        //     int num_exist_edge = name2Edge.at(ename).size();
        //     ename=ename+"/"+std::to_string(num_exist_edge);
        // }
        des1->set(ename, block_op, vertices, "None",0, line_num_block);
        // std::cout<<des1->get_dimension()<<std::endl;
        g.add_edge_to_graph(des1);
        g.add_or_update_vertex(block_op);
        for (size_t i = 0; i < operations.size(); ++i) {
            operations[i]->updateOutOp(block_op,&(g.getGraph()));
        }
    }



    std::ofstream file("output.txt");
    if (file.is_open()) {
        g.printGraph(file);
        file.close();
    } else {
        std::cerr << "Unable to open file" << std::endl;
    }
    inFile.close();
      return 0;
}