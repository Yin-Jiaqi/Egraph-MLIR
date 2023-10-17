#include <iostream>
#include <unordered_map>
#include <functional>
#include <vector>
#include <typeinfo>
#include "utilities.hpp"
#include "graph.hpp"
using MyTupleType1 = std::tuple<std::string, std::string, std::string, std::vector<std::string>, int, std::string>;
using MyTupleType2 = std::tuple<std::string, std::string, std::string, std::vector<std::pair<std::string, std::tuple<int, std::string>>>, std::string>;
using MyTupleType3 = std::tuple<std::string, std::string, std::string, std::vector<std::string>>;
// using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, VertexProperty, EdgeProperty>;

class Dialect {
public:
    // virtual void parse(const std::string& str, MyTupleType1& tpl ,Boost_Graph& g) const = 0;
    // virtual void parse(const std::string& str, MyTupleType2& tpl ,Boost_Graph& g) const = 0;
    virtual ~Dialect() {}
    std::unordered_map<std::string, std::string> DtypeMap = {
    {"index","Index"}, {"i32","I32"}, {"i64","I64"}, {"f32","F32"}, {"f64","F64"}, {"i32 to index","I32Index"}, {"i64 to index","I64Index"} 
    };
        
};

class Arith : public Dialect {
public:
    void parse(const std::string& str, MyTupleType1& tpl ,Boost_Graph& g) const {
        auto it = operationMap.find(std::get<2>(tpl));
        if (it!=operationMap.end()){
            (it->second)(str,tpl,g);
        }
        else{
            std::cout<<"No operation"<<std::endl;
        }
    }

private:

    void parse_arith(const std::string& str, const MyTupleType1& tpl, Boost_Graph& g, const std::string& op_type) const {
        Edge* des1 = new Edge;
        
        auto vector_size = g.getVertexList().find("Arith")->second.find(op_type)->second.size();
        std::vector<Edge*> source_edges;
        
        // Create source edges dynamically based on tpl size
        for (size_t i = 0; i < std::get<3>(tpl).size(); ++i) source_edges.push_back(new Edge);



        Vertex* arith_op = new Vertex(std::string("Arith_") + op_type + "_" + std::to_string(vector_size), 
                                        "Arith", op_type, DtypeMap.at(utilities::last_n_from_tuple<1>(tpl)),
                                        source_edges, 
                                        std::vector<Edge*>{des1}, utilities::last_n_from_tuple<2>(tpl));
        g.add_or_update_vertex(arith_op);
        std::vector<Vertex*> sink_op = std::vector<Vertex*> {g.getGraph()[1]};
        if (std::get<0>(tpl) != ""){
            des1->set(std::get<0>(tpl), arith_op, sink_op, DtypeMap.at(utilities::last_n_from_tuple<1>(tpl)),utilities::last_n_from_tuple<2>(tpl));
            sink_op[0]->update_input(des1);
            g.add_edge_to_graph(des1);
        }
        auto process_edge = [&](Vertex* op, size_t edge_index, const std::string& ename) {
            if (g.edge_exists(ename)) {
                g.update_edge_to_graph(ename, op, edge_index);
            } else {
                std::vector<Vertex*> temp2 = { op };
                Vertex* source_op = g.getGraph()[0];
                Edge* edge = op->get_in()[edge_index];
                edge->set(ename, source_op, temp2, "None", 0);
                g.add_edge_to_graph(edge);
                source_op->update_output(edge);
            }
        };

        auto input_vector=std::get<3>(tpl);

        for (size_t i = 0; i < input_vector.size(); ++i) {
            process_edge(arith_op, i, input_vector.at(i));
        }
    }


    std::unordered_map<std::string, std::function<void(const std::string&, MyTupleType1&, Boost_Graph& g)>> operationMap = {
        {"addi", [this](const std::string& str, const MyTupleType1& tpl, Boost_Graph& g) { this->parse_arith(str, tpl, g, "Addi"); }},
        {"muli", [this](const std::string& str, const MyTupleType1& tpl, Boost_Graph& g) { this->parse_arith(str, tpl, g, "Muli"); }},
        {"constant", [this](const std::string& str, const MyTupleType1& tpl, Boost_Graph& g) { this->parse_arith(str, tpl, g, "Constant"); }},
        {"index_cast", [this](const std::string& str, const MyTupleType1& tpl, Boost_Graph& g) { this->parse_arith(str, tpl, g, "IndexCast"); }}
    };
};

class Memref : public Dialect {
public:
    void parse(const std::string& str, MyTupleType1& tpl ,Boost_Graph& g) const {
        auto it = operationMap.find(std::get<2>(tpl));
        if (it!=operationMap.end()){
            (it->second)(str,tpl,g);
        }
        else{
            std::cout<<"No operation"<<std::endl;
        }
    }

private:

    void parse_memref(const std::string& str, const MyTupleType1& tpl, Boost_Graph& g, const std::string& op_type) const {
        Edge* des1 = new Edge;
        
        auto vector_size = g.getVertexList().find("Memref")->second.find(op_type)->second.size();
        std::vector<Edge*> source_edges;
        
        // Create source edges dynamically based on tpl size
        for (size_t i = 0; i < std::get<3>(tpl).size(); ++i) source_edges.push_back(new Edge);

        Vertex* memref_op = new Vertex(std::string("Memref_") + op_type + "_" + std::to_string(vector_size), 
                                        "Memref", op_type, DtypeMap.at(utilities::last_n_from_tuple<1>(tpl)),
                                        source_edges, 
                                        std::vector<Edge*>{des1}, utilities::last_n_from_tuple<2>(tpl));
        g.add_or_update_vertex(memref_op);
        std::vector<Vertex*> sink_op = std::vector<Vertex*> {g.getGraph()[1]};
        if (std::get<0>(tpl) != ""){
            des1->set(std::get<0>(tpl), memref_op, sink_op, DtypeMap.at(utilities::last_n_from_tuple<1>(tpl)),utilities::last_n_from_tuple<2>(tpl));
            sink_op[0]->update_input(des1);
            g.add_edge_to_graph(des1);
        }
        auto process_edge = [&](Vertex* op, size_t edge_index, const std::string& ename) {
            if (g.edge_exists(ename)) {
                g.update_edge_to_graph(ename, op, edge_index);
            } else {
                std::vector<Vertex*> temp2 = { op };
                Vertex* source_op = g.getGraph()[0];
                Edge* edge = op->get_in()[edge_index];
                edge->set(ename, source_op, temp2, "None", 0);
                g.add_edge_to_graph(edge);
                source_op->update_output(edge);
            }
        };

        auto input_vector=std::get<3>(tpl);

        for (size_t i = 0; i < input_vector.size(); ++i) {
            process_edge(memref_op, i, input_vector.at(i));
        }
    }


    std::unordered_map<std::string, std::function<void(const std::string&, MyTupleType1&, Boost_Graph& g)>> operationMap = {
        {"load", [this](const std::string& str, const MyTupleType1& tpl, Boost_Graph& g) { this->parse_memref(str, tpl, g, "Load"); }},
        {"store", [this](const std::string& str, const MyTupleType1& tpl, Boost_Graph& g) { this->parse_memref(str, tpl, g, "Store"); }}
    };
};



class Func : public Dialect {
public:
    Vertex* parse(const std::string& str, MyTupleType2& tpl ,Boost_Graph& g) const {
        auto it = operationMap.find(std::get<1>(tpl));
        if (it!=operationMap.end()){
            return (it->second)(str,tpl,g);
        }
        else{
            std::cout<<"No operation"<<std::endl;
            return nullptr;
        }
    }

private:

    Vertex* parse_func(const std::string& str, const MyTupleType2& tpl, Boost_Graph& g, const std::string& op_type) const {

        Edge* des1 = new Edge;
        auto vector_size = g.getVertexList().find("Func")->second.find(op_type)->second.size();

        std::vector<Edge*> source_edges;
        
        // Create source edges dynamically based on tpl size
        for (size_t i = 0; i < std::get<3>(tpl).size(); ++i) source_edges.push_back(new Edge);

        Vertex* func_op = new Vertex(std::string("Func_") + op_type + "_" + std::to_string(vector_size), 
                                        "Func", op_type, "None",
                                        source_edges, 
                                        std::vector<Edge*>{des1}, 0);
        g.add_or_update_vertex(func_op);
        std::vector<Vertex*> sink_op = std::vector<Vertex*> {g.getGraph()[1]};
        auto process_edge = [&](Vertex* op, size_t edge_index, const std::pair<std::string, std::tuple<int, std::string>>& e) {
            if (g.edge_exists(e.first)) {
                g.update_edge_to_graph(e.first, op, edge_index);
            } else {
                std::vector<Vertex*> temp2 = { op };
                Vertex* source_op = g.getGraph()[0];
                Edge* edge = op->get_in()[edge_index];
                edge->set(e.first, source_op, temp2, DtypeMap.at(std::get<1>(e.second)), std::get<0>(e.second));
                g.add_edge_to_graph(edge);
                source_op->update_output(edge);
            }
        };

        auto input_vector=std::get<3>(tpl);

        for (size_t i = 0; i < input_vector.size(); ++i) {
            process_edge(func_op, i, input_vector.at(i));
        }

        return func_op;
    }

    std::unordered_map<std::string, std::function<Vertex*(const std::string&, MyTupleType2&, Boost_Graph& g)>> operationMap = {
        {"func", [this](const std::string& str, const MyTupleType2& tpl, Boost_Graph& g) { return this->parse_func(str, tpl, g, "Func"); }},
        {"return", [this](const std::string& str, const MyTupleType2& tpl, Boost_Graph& g) { return this->parse_func(str, tpl, g, "Return"); }}
    };
};


class Scf : public Dialect {
public:
    Vertex* parse(const std::string& str, MyTupleType3& tpl ,Boost_Graph& g) const {
        auto it = operationMap.find(std::get<1>(tpl));
        if (it!=operationMap.end()){
            return (it->second)(str,tpl,g);
        }
        else{
            std::cout<<"No operation"<<std::endl;
            return nullptr;
        }
    }

private:

    Vertex* parse_scf(const std::string& str, const MyTupleType3& tpl, Boost_Graph& g, const std::string& op_type) const {
        Edge* des1 = new Edge;
        auto vector_size = g.getVertexList().find("Scf")->second.find(op_type)->second.size();

        std::vector<Edge*> source_edges;
        
        // Create source edges dynamically based on tpl size
        for (size_t i = 0; i < std::get<3>(tpl).size(); ++i) source_edges.push_back(new Edge);
        Vertex* scf_op = new Vertex(std::string("Scf_") + op_type + "_" + std::to_string(vector_size), 
                                        "Scf", op_type, "None",
                                        source_edges, 
                                        std::vector<Edge*>{des1}, 0);
        g.add_or_update_vertex(scf_op);
        std::vector<Vertex*> sink_op = std::vector<Vertex*> {g.getGraph()[1]};
        if (std::get<2>(tpl) != ""){
            des1->set(std::get<2>(tpl), scf_op, sink_op, "None",0);
            sink_op[0]->update_input(des1);
            g.add_edge_to_graph(des1);
        }

        auto process_edge = [&](Vertex* op, size_t edge_index, const std::string& e) {
            if (g.edge_exists(e)) {
                g.update_edge_to_graph(e, op, edge_index);
            } else {
                std::vector<Vertex*> temp2 = { op };
                Vertex* source_op = g.getGraph()[0];
                Edge* edge = op->get_in()[edge_index];
                edge->set(e, source_op, temp2, "None", 0);
                g.add_edge_to_graph(edge);
                source_op->update_output(edge);
            }
        };

        auto input_vector=std::get<3>(tpl);

        for (size_t i = 0; i < input_vector.size(); ++i) {
            process_edge(scf_op, i, input_vector.at(i));
        }

        return nullptr;
    }

    std::unordered_map<std::string, std::function<Vertex*(const std::string&, MyTupleType3&, Boost_Graph& g)>> operationMap = {
        {"for", [this](const std::string& str, const MyTupleType3& tpl, Boost_Graph& g) { return this->parse_scf(str, tpl, g, "For"); }},
    };
};
