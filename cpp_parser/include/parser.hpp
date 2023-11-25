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
    std::unordered_map<std::string, std::string> DialectMap = {
    {"arith","Arith"}, {"affine","Affine"}, {"func","Func"}, {"scf","Scf"}, {"memref","Memref"}
    };

    static std::tuple<int, std::string>
    analyzeType(const std::string& type_str) {
        auto main_parts = utilities::split(type_str.substr(1), "<>");
        if (main_parts.size() != 1 && main_parts.size() != 2) {
            std::cerr << "Error: Invalid MLIR type format." << std::endl;
            throw std::runtime_error("Invalid MLIR format");
        }
        
        if (main_parts.size() == 1) {
            return {0, main_parts[0]};  // Not a tensor type, just return data type
        } else {
            auto type_parts = utilities::split(main_parts[1], "x");
            int dimensions = type_parts.size() - 1;
            std::string dataType =  type_parts.back();
            return {dimensions, dataType};
        }
    }

    static std::tuple<std::string, std::string, std::string>
    splitPrimary(const std::string& expression) {
        auto parts = utilities::split(expression, "=");
        if (parts.size() != 1 && parts.size() != 2) {
            std::cerr << "Error: Invalid MLIR expression format." << std::endl;
            throw std::runtime_error("Invalid MLIR format1");
        }
        
        std::string outputValue = (parts.size() == 2) ? utilities::split(parts[0]," ")[0] : "Pseudo";
        std::string toParse = (parts.size() == 2) ? parts[1] : parts[0];
        
        auto secondary_parts = utilities::split(toParse, ":");
        if (secondary_parts.size() != 2) {
            std::cerr << "Error: Invalid MLIR expression format." << std::endl;
            throw std::runtime_error("Invalid MLIR format2");
        }
        return {outputValue, secondary_parts[0], secondary_parts[1]};
    }

    static std::tuple<std::string, std::string, std::vector<std::string>>
    analyzeOperation(const std::string& operation_str) {
        auto operation_parts = utilities::split(operation_str, ". ,[]");
        if (operation_parts.size() < 2) {
            std::cerr << "Error: Invalid MLIR expression format." << std::endl;
            throw std::runtime_error("Invalid MLIR format3");
        }
        std::string dialect = operation_parts[0];
        std::string operation = operation_parts[1];
        std::vector<std::string> parameters(operation_parts.begin() + 2, operation_parts.end());
        // two corner case
        // %cst = arith.constant 2.000000e-01 : f64
	    // %3 = affine.load %arg2[%arg5, %arg6 - 1] : memref<?x1300xf64>
        return {dialect, operation, parameters};
    }



    static MyTupleType1 analyzeMLIR(const std::string& expression,
                std::function<std::tuple<std::string, std::string, std::string>(const std::string&)> splitPrimaryFn = splitPrimary,
                std::function<std::tuple<std::string, std::string, std::vector<std::string>>(const std::string&)> analyzeOperationFn = analyzeOperation,
                std::function<std::tuple<int, std::string>(const std::string&)> analyzeTypeFn = analyzeType) {
        auto [output, operation_str, type_str] = splitPrimaryFn(expression);
        auto [dialect, operation, parameters] = analyzeOperationFn(operation_str);
        auto [dimensions, dataType] = analyzeTypeFn(type_str);
        return {output, dialect, operation, parameters, dimensions, dataType};
    }
};

class Computation : public Dialect {
public:
    void parse(const std::string& str, MyTupleType1& tpl ,Boost_Graph& g, int line_num, std::vector<Block*> todo_blist) const {
        auto it = operationMap.find(std::get<2>(tpl));
        if (it!=operationMap.end()){
            (it->second)(str,tpl,g,line_num,todo_blist);
        }
        else{
            throw std::runtime_error("No operation:"+std::get<2>(tpl));
        }
    }

private:

    void parse_computation(const std::string& str, const MyTupleType1& tpl, Boost_Graph& g, const std::string& op_type, int line_num, std::vector<Block*> todo_blist) const {
        Edge* des1 = new Edge;
        auto dialect=std::get<1>(tpl);
        auto dialect_name= DialectMap.at(dialect);
        auto vector_size = g.getVertexList().find(dialect_name)->second.find(op_type)->second.size();
        std::vector<Edge*> source_edges;
        // Create source edges dynamically based on tpl size
        for (size_t i = 0; i < std::get<3>(tpl).size(); ++i) source_edges.push_back(new Edge);
        if (todo_blist.empty()){
            source_edges.push_back(new Edge);
        }
        Vertex* computation_op = new Vertex(std::string(dialect_name+"_") + op_type + "_" + std::to_string(vector_size), 
                                        dialect_name, op_type, DtypeMap.at(utilities::last_n_from_tuple<1>(tpl)),
                                        source_edges, 
                                        std::vector<Edge*>{des1}, utilities::last_n_from_tuple<2>(tpl), line_num, todo_blist.size());
        g.add_or_update_vertex(computation_op);
        std::vector<Vertex*> sink_op = std::vector<Vertex*> {g.getGraph()[0]};
        auto ename = std::get<0>(tpl);
        if (ename != ""){
            if (g.edge_exists_l2(ename)){
                auto name2Edge = g.getname2Edge();
                auto e = name2Edge.at(ename).at(ename);
                e->setEdgeName(ename+"/0");
                name2Edge[ename].erase(ename);
                name2Edge[ename][ename+"/0"] = e;
                ename=ename+"/1";
            }
            else if (g.edge_exists_l2(ename+"/0")){
                auto name2Edge = g.getname2Edge();
                int num_exist_edge = name2Edge.at(ename).size();
                ename=ename+"/"+std::to_string(num_exist_edge);
            }
            des1->set(ename, computation_op, sink_op, DtypeMap.at(utilities::last_n_from_tuple<1>(tpl)),utilities::last_n_from_tuple<2>(tpl), line_num);
            sink_op[0]->update_input(des1);
            g.add_edge_to_graph(des1);
        }
        auto process_edge = [&](Vertex* op, size_t edge_index, const std::string& ename) {
            if (g.edge_exists_l1(ename)) {
                auto ename_true = g.getname2Edge().at(ename).rbegin()->first;
                g.update_edge_to_graph(ename_true, op, edge_index);
            } else {
                std::vector<Vertex*> temp2 = { op };
                auto vector_size = g.getVertexList().find("SS")->second.find("Source")->second.size();
                Vertex* source_op = new Vertex("Source_" + std::to_string(vector_size),"SS","Source","None",{}, {},0, -1, 0);
                g.add_or_update_vertex(source_op);
                Edge* edge = op->get_in()[edge_index];
                edge->set(ename, source_op, temp2, "None", 0, -1);
                g.add_edge_to_graph(edge);
                source_op->update_output(edge);
            }
        };

        auto input_vector=std::get<3>(tpl);

        for (size_t i = 0; i < input_vector.size(); ++i) {
            process_edge(computation_op, i, input_vector.at(i));
        }


        if (!todo_blist.empty()){
            todo_blist.back()->addOperation(des1);
        }
        else{
            g.update_edge_to_graph("No_block", computation_op, input_vector.size());
        }
    }


    std::unordered_map<std::string, std::function<void(const std::string&, MyTupleType1&, Boost_Graph& g, int line_num, std::vector<Block*> todo_blist)>> operationMap = {
        {"addi", [this](const std::string& str, const MyTupleType1& tpl, Boost_Graph& g, int line_num, std::vector<Block*> todo_blist) { this->parse_computation(str, tpl, g, "Addi", line_num, todo_blist); }},
        {"addf", [this](const std::string& str, const MyTupleType1& tpl, Boost_Graph& g, int line_num, std::vector<Block*> todo_blist) { this->parse_computation(str, tpl, g, "Addf", line_num, todo_blist); }},
        {"muli", [this](const std::string& str, const MyTupleType1& tpl, Boost_Graph& g, int line_num, std::vector<Block*> todo_blist) { this->parse_computation(str, tpl, g, "Muli", line_num, todo_blist); }},
        {"mulf", [this](const std::string& str, const MyTupleType1& tpl, Boost_Graph& g, int line_num, std::vector<Block*> todo_blist) { this->parse_computation(str, tpl, g, "Mulf", line_num, todo_blist); }},
        {"constant", [this](const std::string& str, const MyTupleType1& tpl, Boost_Graph& g, int line_num, std::vector<Block*> todo_blist) { this->parse_computation(str, tpl, g, "Constant", line_num, todo_blist); }},
        {"index_cast", [this](const std::string& str, const MyTupleType1& tpl, Boost_Graph& g, int line_num, std::vector<Block*> todo_blist) { this->parse_computation(str, tpl, g, "IndexCast", line_num, todo_blist); }},
        {"load", [this](const std::string& str, const MyTupleType1& tpl, Boost_Graph& g, int line_num, std::vector<Block*> todo_blist) { this->parse_computation(str, tpl, g, "Load", line_num, todo_blist); }},
        {"store", [this](const std::string& str, const MyTupleType1& tpl, Boost_Graph& g, int line_num, std::vector<Block*> todo_blist) { this->parse_computation(str, tpl, g, "Store", line_num, todo_blist); }}
    };
};


class Func : public Dialect {
public:
    Vertex* parse(const std::string& str, MyTupleType2& tpl ,Boost_Graph& g, int line_num, std::vector<Block*> todo_blist) const {
        auto it = operationMap.find(std::get<1>(tpl));
        if (it!=operationMap.end()){
            return (it->second)(str,tpl,g,line_num,todo_blist);
        }
        else{
            throw std::runtime_error("No operation:"+std::get<2>(tpl));
        }
    }


    static std::tuple<std::string, std::string, std::string, std::vector<std::pair<std::string, std::tuple<int, std::string>>>, std::string>
    split_data(const std::string &input) {
        auto parts = utilities::split(input, "{}");
        auto main_parts = utilities::split(parts[0], "().@");

        if (main_parts.size() != 5) {
            std::cerr << "Error: Invalid MLIR type format." << std::endl;
            throw std::runtime_error("Invalid MLIR format");
        }


        std::string dialect = utilities::split(main_parts[0], " ")[0];
        std::string operation = utilities::split(main_parts[1], " ")[0];
        std::string function_name = utilities::split(main_parts[2], " ")[0];
        std::string params_str = main_parts[3];
        std::string attributes = main_parts[4] + "{" + parts[1] + "}";

        auto params_list = utilities::split(params_str, ",");
        std::vector<std::pair<std::string, std::tuple<int, std::string>>> params;

        for (const auto &param_str : params_list) {
            auto param_parts = utilities::split(param_str, ":");
            if (param_parts.size() != 2) {
                std::cerr << "Error: Invalid parameter format.2" << std::endl;
                throw std::runtime_error("Invalid parameter format");
            }
            std::string param_name = utilities::split(param_parts[0]," ")[0];
            auto param_type = analyzeType(param_parts[1]);
            params.emplace_back(param_name, param_type);
        }

        return {dialect, operation, function_name, params, attributes};
    }

private:

    Vertex* parse_func(const std::string& str, const MyTupleType2& tpl, Boost_Graph& g, const std::string& op_type, int line_num, std::vector<Block*> todo_blist) const {

        Edge* des1 = new Edge;
        auto vector_size = g.getVertexList().find("Func")->second.find(op_type)->second.size();

        std::vector<Edge*> source_edges;
        
        // Create source edges dynamically based on tpl size
        for (size_t i = 0; i < std::get<3>(tpl).size(); ++i) source_edges.push_back(new Edge);
        if (todo_blist.empty()){
            source_edges.push_back(new Edge);
        }
        Vertex* func_op = new Vertex(std::string("Func_") + op_type + "_" + std::to_string(vector_size), 
                                        "Func", op_type, "None",
                                        source_edges, 
                                        std::vector<Edge*>{des1}, 0, line_num, todo_blist.size());
        g.add_or_update_vertex(func_op);
        std::vector<Vertex*> sink_op = std::vector<Vertex*> {g.getGraph()[0]};
        auto process_edge = [&](Vertex* op, size_t edge_index, const std::pair<std::string, std::tuple<int, std::string>>& e) {
            if (g.edge_exists_l1(e.first)) {
                throw std::runtime_error("The input should not exist:"+e.first);
            } else {
                std::vector<Vertex*> temp2 = { op };
                auto vector_size = g.getVertexList().find("SS")->second.find("Source")->second.size();
                Vertex* source_op = new Vertex("Source_" + std::to_string(vector_size),"SS","Source",DtypeMap.at(std::get<1>(e.second)),{}, {},std::get<0>(e.second), -1, 0);
                g.add_or_update_vertex(source_op);
                Edge* edge = op->get_in()[edge_index];
                edge->set(e.first, source_op, temp2, DtypeMap.at(std::get<1>(e.second)), std::get<0>(e.second), line_num);
                g.add_edge_to_graph(edge);
                source_op->update_output(edge);
            }
        };

        auto input_vector=std::get<3>(tpl);

        
        for (size_t i = 0; i < input_vector.size(); ++i) {
            process_edge(func_op, i, input_vector.at(i));
        }

        if (!todo_blist.empty()){
            todo_blist.back()->addOperation(des1);
        }
        else{
            g.update_edge_to_graph("No_block", func_op, input_vector.size());
        }


        return func_op;
    }

    std::unordered_map<std::string, std::function<Vertex*(const std::string&, MyTupleType2&, Boost_Graph& g, int line_num, std::vector<Block*> todo_blist)>> operationMap = {
        {"func", [this](const std::string& str, const MyTupleType2& tpl, Boost_Graph& g, int line_num, std::vector<Block*> todo_blist) { return this->parse_func(str, tpl, g, "Func", line_num, todo_blist); }},
        {"return", [this](const std::string& str, const MyTupleType2& tpl, Boost_Graph& g, int line_num, std::vector<Block*> todo_blist) { return this->parse_func(str, tpl, g, "Return", line_num, todo_blist); }}
    };
};


class For : public Dialect {
public:
    Vertex* parse_for(const std::string& str, MyTupleType3& tpl ,Boost_Graph& g, int line_num, std::vector<Block*> todo_blist, const std::string& output_type) const {
        auto it = operationMap.find(std::get<1>(tpl));
        if (it!=operationMap.end()){
            return (it->second)(str,tpl,g,line_num,todo_blist);
        }
        else{
            throw std::runtime_error("No operation:"+std::get<2>(tpl));
        }
    }

    static std::tuple<std::string, std::string, std::string, std::vector<std::string>>
    split_data(const std::string &input) {
        std::vector<std::string> splitResult = utilities::split(input, "=");

        if (splitResult.size() != 2) {
            std::cout << "Error: Invalid MLIR type format." << std::endl;
            throw std::runtime_error("Invalid MLIR format");
        }

        std::string returnValue = splitResult[0];
        std::string loopParameter = splitResult[1];

        std::vector<std::string> returnValueSplit = utilities::split(returnValue, ". ");
        if (returnValueSplit.size() != 3) {
            std::cout << "Error: Invalid MLIR type format." << std::endl;
            throw std::runtime_error("Invalid MLIR format");
        }

        std::string dialect = returnValueSplit[0];
        std::string operation = returnValueSplit[1];
        std::string retValue = returnValueSplit[2];

        std::vector<std::string> loopParameterSplit = utilities::split(loopParameter, " ");
        if (loopParameterSplit.size()!= 4 && loopParameterSplit.size()!= 6 || loopParameterSplit[1]!="to" || utilities::last_n_from_vector(loopParameterSplit,1)!="{"){
            std::cout << "Error: Invalid MLIR type format." << std::endl;
            throw std::runtime_error("Invalid MLIR format");
        }

        std::vector<std::string> selectedLoopParameters;
        for (size_t i = 0; i < loopParameterSplit.size(); i += 2) {
            selectedLoopParameters.push_back(loopParameterSplit[i]);
        }

        if (selectedLoopParameters.size()==2){
            selectedLoopParameters.push_back("1");
        }

        return std::make_tuple(dialect, operation, retValue, selectedLoopParameters);
    }

private:

    Vertex* parse_for(const std::string& str, const MyTupleType3& tpl, Boost_Graph& g, const std::string& op_type, int line_num, std::vector<Block*> todo_blist) const {
        Edge* des1 = new Edge;
        auto dialect=std::get<0>(tpl);
        auto dialect_name= DialectMap.at(dialect);
        auto vector_size_value = g.getVertexList().find(dialect_name)->second.find(op_type+ "value")->second.size();

        std::vector<Edge*> source_edges_value;
        for (size_t i = 0; i < std::get<3>(tpl).size(); ++i) source_edges_value.push_back(new Edge);
        if (todo_blist.empty()){
            source_edges_value.push_back(new Edge);
        }
        Vertex* for_op_value = new Vertex;
        for_op_value->set(std::string(dialect_name+"_") + op_type + "value_" + std::to_string(vector_size_value), 
                                        dialect_name, op_type+ "value", "None",
                                        source_edges_value,
                                        std::vector<Edge*>{des1}, 0, line_num, todo_blist.size());
        g.add_or_update_vertex(for_op_value);
        std::vector<Vertex*> sink_op = std::vector<Vertex*> {g.getGraph()[0]};
        auto ename_value = std::get<2>(tpl);
        if (g.edge_exists_l2(ename_value)){
            auto name2Edge = g.getname2Edge();
            auto e = g.getname2Edge().at(ename_value).at(ename_value);
            e->setEdgeName(ename_value+"/0");
            name2Edge.at(ename_value).erase(ename_value);
            g.getname2Edge().at(ename_value).erase(ename_value);
            g.getname2Edge().at(ename_value)[ename_value+"/0"] = e;
            ename_value=ename_value+"/1";
        }
        else if (g.edge_exists_l2(ename_value+"/0")){
            auto name2Edge = g.getname2Edge();
            int num_exist_edge = name2Edge.at(ename_value).size();
            ename_value=ename_value+"/"+std::to_string(num_exist_edge);
        }
        des1->set(ename_value, for_op_value, sink_op, "None", 0, line_num);
        sink_op[0]->update_input(des1);
        g.add_edge_to_graph(des1);

        auto process_edge = [&](Vertex* op, size_t edge_index, const std::string& ename) {
            if (g.edge_exists_l1(ename)) {
                auto ename_true = g.getname2Edge().at(ename).rbegin()->first;
                g.update_edge_to_graph(ename_true, op, edge_index);
            } else {
                std::vector<Vertex*> temp2 = { op };
                auto vector_size = g.getVertexList().find("SS")->second.find("Source")->second.size();
                Vertex* source_op = new Vertex("Source_" + std::to_string(vector_size),"SS","Source","None",{}, {},0, -1, 0);
                g.add_or_update_vertex(source_op);
                Edge* edge = op->get_in()[edge_index];
                edge->set(ename, source_op, temp2, "None", 0, -1);
                g.add_edge_to_graph(edge);
                source_op->update_output(edge);
            }
        };
        auto input_vector=std::get<3>(tpl);
        for (size_t i = 0; i < input_vector.size(); ++i) {
            process_edge(for_op_value, i, input_vector.at(i));
        }

        Edge* des2 = new Edge;
        Vertex* for_op_control = new Vertex;
        auto vector_size_control = g.getVertexList().find(dialect_name)->second.find(op_type+ "control")->second.size();
        std::vector<Edge*> source_edges_control;source_edges_control.push_back(des1);
        for_op_control->set(std::string(dialect_name+"_") + op_type + "control_" + std::to_string(vector_size_control), 
                                        dialect_name, op_type+ "control", "None",
                                        source_edges_control, 
                                        std::vector<Edge*>{des2}, 0, line_num, todo_blist.size());
        g.add_or_update_vertex(for_op_control);
        std::string ename_control = "Pseudo";
        if (g.edge_exists_l2(ename_control)){
            auto name2Edge = g.getname2Edge();
            auto e = g.getname2Edge().at(ename_control).at(ename_control);
            e->setEdgeName(ename_control+"/0");
            name2Edge.at(ename_control).erase(ename_control);
            g.getname2Edge().at(ename_control).erase(ename_control);
            g.getname2Edge().at(ename_control)[ename_control+"/0"] = e;
            ename_control=ename_control+"/1";
        }
        else if (g.edge_exists_l2(ename_control+"/0")){
            auto name2Edge = g.getname2Edge();
            int num_exist_edge = name2Edge.at(ename_control).size();
            ename_control=ename_control+"/"+std::to_string(num_exist_edge);
        }
        des2->set(ename_control, for_op_control, sink_op, "None", 0, line_num);
        sink_op[0]->update_input(des2);
        des1->updateOutOp(for_op_control, &(g.getGraph()));
        g.add_edge_to_graph(des2);

        todo_blist.back()->addOperation(des2);

        return for_op_control;
    }

    std::unordered_map<std::string, std::function<Vertex*(const std::string&, MyTupleType3&, Boost_Graph& g, int line_num, std::vector<Block*> todo_blist)>> operationMap = {
        {"for", [this](const std::string& str, const MyTupleType3& tpl, Boost_Graph& g, int line_num, std::vector<Block*> todo_blist) { return this->parse_for(str, tpl, g, "For", line_num, todo_blist); }},
    };
};
