#include "thorin/analyses/cgra_vector_width.h"

#include <unordered_set>

namespace thorin {

template<bool allow_scaling>
size_t CGRAVectorWidthAnalysis<allow_scaling>::get_width_for (const Continuation* cont) {
    using DeviceApiSet = std::unordered_set<std::string>;
    DeviceApiSet irregular_apis = { "aie::vector::extract", "aie::zeros", "aie::store_v", "readincr_v_channel", "window_readincr_v_channel", "aie::load_v", "aie::sliding_"/*all sliding APIs*/, "srs"};

    auto new_vector_size = base_size;

    for (auto use: cont->uses()) {
        if (auto app = use->isa<App>(); app && app->callee()->isa_nom<Continuation>()) {
            auto callee = app->callee()->as_nom<Continuation>();
            if (callee->cc() == CC::Device) {
                auto name = callee->name();
                if (name.find("aie::sliding_") != std::string::npos)
                    name = "aie::sliding_";
                if (irregular_apis.count(name)) {
                    if (app->num_args() > 1) {
                        // The first arg of all irregular APIs is the lane size
                        if (auto primtype = app->arg(1)->type()->isa<PrimType>()) {
                            if (primtype->primtype_tag() == PrimType_pu32) {
                                new_vector_size = app->arg(1)->as<PrimLit>()->value().get_u32();
                            } else {
                                world().WLOG("Lane size in {} must be an unsigned integer value to be effective", name);
                            }
                        }
                    }
                }
            }
        }
    }

    return new_vector_size;
}

template<bool allow_scaling>
size_t CGRAVectorWidthAnalysis<allow_scaling>::get_width_for (const Def* def) {
    if (auto width = def_2_size.lookup(def))
        return *width;

    size_t vector_width = base_size;

    for (auto use : def->uses()) {
        if (auto cont = use->isa<Continuation>()) {
            auto width = get_width_for(cont);
            if (allow_scaling) {
                if (width > vector_width)
                    vector_width = width;
            } else {
                assert(vector_width == base_size || vector_width == width);
                vector_width = width;
            }
        }
    }

    return vector_width;
}

template<bool allow_scaling>
void CGRAVectorWidthAnalysis<allow_scaling>::register_width (const Def* def, size_t width) {
    def_2_size.emplace(def, width);
}

template class CGRAVectorWidthAnalysis<true>;
//template class CGRAVectorWidthAnalysis<false>;

}
