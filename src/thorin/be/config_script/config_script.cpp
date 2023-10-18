#include "config_script.h"
#include "thorin/analyses/scope.h"


namespace thorin::config_script {

// check if a hls param is a left-hand-side parameter
bool is_a_lhs(ChannelMode hls_mode) {
    return hls_mode == ChannelMode::Write;
}

static inline std::string create_hls_kernel(std::string top_name, size_t num_instance) {
    auto fixed_part = "nk = "+ top_name + ":" + std::to_string(num_instance) + ":";
    auto single_instance = fixed_part + top_name + "\n";
    std::string instance_names;
    for (size_t i = 0 ; i < num_instance; ++i) {
        instance_names += top_name + "_" + std::to_string(i);
        if (i < num_instance - 1 )
            instance_names += ":";
    }
    auto multiple_instance = fixed_part + instance_names + "\n";
    return num_instance == 1 ? (single_instance + "\n") : (multiple_instance + "\n");

}

void emit_stream_connect(Stream& stream, Ports hls_cgra_ports) {

    //lambdas over this loop
    for (const auto& ports : hls_cgra_ports) {
        auto [hls_param2mode, cgra_param] = ports;
        auto [hls_param, hls_mode] = hls_param2mode.value();

        auto is_a_hls = [&] (const Def* param) {
            return hls_mode == ChannelMode::Write;
        };


        auto append_to_hls = [&] (std::string top_name, const Def* param) {
            return top_name + "." + param->unique_name();
        };

        auto append_to_cgra = [&] (const Def* param) {
            return "ai_engine_0." + param->unique_name();
        };
        auto hls_param_name = append_to_hls("hls_top", hls_param);
        auto cgra_param_name = append_to_cgra(cgra_param.value());

        std::string lhs, rhs;
        if (is_a_lhs(hls_mode)) {
            lhs = hls_param_name;
            rhs  = cgra_param_name;
        } else {
            lhs = cgra_param_name;
            rhs = hls_param_name;
        }

        stream << "stream_connect = " << lhs << ":" << rhs << "\n";

    }
}


void emit_system_port_tag(Stream& stream) {/*TODO: for memory port/bank assignments*/}

void tool_config(std::string& flags, Stream& stream) {

    stream << "[vivado]" << "\n";
    bool fast_emu = false;
    if (!flags.empty()) {
        for (auto& ch : flags)
            ch = std::toupper(ch, std::locale());

        std::istringstream flags_stream(flags);
        std::string token;

        while (std::getline(flags_stream, token, ',')) {
            if (token.compare("FAST_EMU") == 0) {
                fast_emu = true;
                break;
            } else {
                continue;
            }
        }
    }

    if (fast_emu) {
    stream << "prop=fileset.sim_1.xsim.elaborate.xelab.more_options={-override_timeprecision -timescale=1ns/1ps}\n";
    }

}

Stream&  emit_connectivity(Stream& stream, Ports hls_cgra_ports) {
    /*all above inside this one*/
    if (hls_cgra_ports.empty())
        return stream;
    stream << "[connectivity]" << "\n" ;
    stream << create_hls_kernel("hls_top", 1);
    emit_stream_connect(stream, hls_cgra_ports);
    return stream;
}

void CodeGen::emit_stream(std::ostream& stream) {

    Stream s(stream);
    emit_connectivity(s, hls_cgra_ports_) << "\n";
    tool_config(flags_,s);

}

}

