#ifndef THORIN_UTILS_H
#define THORIN_UTILS_H


#include "thorin/be/graphs.h"

#include <cassert>
#include <iostream>
#include <iterator>
#include <vector>

namespace thorin {

class ArgsIterator : public std::iterator<std::forward_iterator_tag, const char*> {
public:
    ArgsIterator(int pos, char** argv)
        : pos_(pos)
        , argv_(argv)
    {}

    const char* operator*() const { return argv_[pos_]; }
    ArgsIterator& operator++() { ++pos_; return *this; }
    ArgsIterator operator++(int) { ArgsIterator tmp(pos_, argv_); ++pos_; return tmp; }
    bool operator==(ArgsIterator b) const { return pos_ == b.pos_; }
    bool operator!=(ArgsIterator b) const { return pos_ != b.pos_; }

private:
    int pos_;
    char** argv_;
};

template<typename T>
class BasicOption {
public:
    typedef ArgsIterator iterator;

    BasicOption(const std::string& arg, const std::string& help_msg, T* target)
        : target_(target)
        , arg_(arg)
        , help_msg_(help_msg)
    {}

    T* target() const { return target_; }
    const std::string& arg() const { return arg_; }
    const std::string& help_msg() const { return help_msg_; }

protected:
    T* target_;
    std::string arg_;
    std::string help_msg_;
};

template<typename T, typename Data>
struct Option;

template<typename Data, typename T, typename Class>
class OptionBase : public BasicOption<Data> {
public:
    OptionBase(const T& previous, const std::string& arg, const std::string& help_msg, Data* target, const Data& def)
        : BasicOption<Data>(arg, help_msg, target)
        , previous_(previous)
    {
        *target = def;
    }

    bool is_arg(const char* value) const {
        // TODO: nicer
        if (value[0] != '-')
            return false;

        // allow also --
        int offset = 1;
        if (value[1] == '-') offset++;

        return strcmp(BasicOption<Data>::arg().c_str(), value + offset) == 0;
    }

    typename BasicOption<Data>::iterator handle(typename BasicOption<Data>::iterator it) const {
        const Class* c = as_class();
        if (!*it)
            return it;
        if (c->is_arg(*it))
            return c->handle_option(it);
        return previous_.handle(it);
    }

    const Class* as_class() const { return static_cast<const Class*>(this); }

    template<typename NData>
    Option<NData, Class> add_option(const std::string& arg, const std::string& help_msg, NData& target, const NData& def = NData()) const {
        return Option<NData, Class>(*as_class(), arg, help_msg, &target, def);
    }

    void parse(int argc, char** argv) const {
        // parse command line arguments
        typedef typename BasicOption<Data>::iterator iter;
        for (iter it(1, argv), e(argc, argv); it != e;) {
            auto new_it = handle(it);
            assert(it != new_it);
            it = new_it;
        }
    }

    void print_help() {
        // print some help infos
        const Class* c = as_class();
        previous_.print_help();
        std::cout << "\t-" << c->arg() << "\t" << c->help_msg() << std::endl;
    }

protected:
    T previous_;
};

template<typename Data, typename T>
struct Option : public OptionBase<Data, T, Option<Data, T>> {
    Option(const T& previous, const std::string& arg, const std::string& help_msg, Data* target, const Data& def = Data())
        : OptionBase<Data, T, Option<T, Data>>(previous, arg, help_msg, target, def)
    {}

    typename OptionBase<Data, T, Option<T, Data>>::iterator handle_option(typename OptionBase<Data, T, Option<T, Data>>::iterator it) const {
        assert(false && "illegal use of program option");
    }
};

template<typename T>
struct Option<int, T> : public OptionBase<int, T, Option<int, T>> {
    typedef Option<int, T> self;

    Option(const T& previous, const std::string& arg, const std::string& help_msg, int* target, int def)
        : OptionBase<int, T, self>(previous, arg, help_msg, target, def)
    {}

    typename OptionBase<int, T, self>::iterator handle_option(typename OptionBase<int, T, self>::iterator it) const {
        *OptionBase<int, T, self>::target() = atoi(*++it);
        return ++it;
    }
};

template<typename T>
struct Option<bool, T> : public OptionBase<bool, T, Option<bool, T>> {
    typedef Option<bool, T> self;

    Option(const T& previous, const std::string& arg, const std::string& help_msg, bool* target, bool def)
        : OptionBase<bool, T, self>(previous, arg, help_msg, target, def)
    {}

    typename OptionBase<bool, T, self>::iterator handle_option(typename OptionBase<bool, T, self>::iterator it) const {
        *OptionBase<bool, T, self>::target() = true;
        return ++it;
    }

};

template<typename T>
struct Option<std::string, T> : public OptionBase<std::string, T, Option<std::string, T>> {
    typedef Option<std::string, T> self;

    Option(const T& previous, const std::string& arg, const std::string& help_msg, std::string* target, const std::string& def)
        : OptionBase<std::string, T, self>(previous, arg, help_msg, target, def)
    {}

    typename OptionBase<std::string, T, self>::iterator handle_option(typename OptionBase<std::string, T, self>::iterator it) const {
        *OptionBase<std::string, T, self>::target() = *++it;
        return ++it;
    }
};

typedef std::vector<std::string> OptionStringVector;

template<typename T>
struct Option<OptionStringVector, T> : public OptionBase<OptionStringVector, T, Option<OptionStringVector, T>> {
    typedef Option<OptionStringVector, T> self;

    Option(const T& previous, const std::string& arg, const std::string& help_msg, OptionStringVector* target, const OptionStringVector& def)
        : OptionBase<OptionStringVector, T, self>(previous, arg, help_msg, target, def)
    {}

    typename OptionBase<OptionStringVector, T, self>::iterator handle_option(typename OptionBase<OptionStringVector, T, self>::iterator it) const {
        // it points to the current argument
        // -> skip it
        ++it;
        do
        {
            OptionBase<OptionStringVector, T, self>::target()->push_back(*it++);
        } while (*it && (*it)[0] != '-');
        return it;
    }
};

template<typename T>
struct Option<YCompCommandLine, T> : public OptionBase<YCompCommandLine, T, Option<YCompCommandLine, T>> {
    typedef Option<YCompCommandLine, T> self;

    Option(const T& previous, const std::string& arg, const std::string& help_msg, YCompCommandLine* target, const YCompCommandLine& def)
            : OptionBase<YCompCommandLine, T, self>(previous, arg, help_msg, target, def)
    {}

    bool has_next(typename OptionBase<YCompCommandLine, T, self>::iterator it) const {
        return *it && (*it)[0] != '-';
    }

    typename OptionBase<YCompCommandLine, T, self>::iterator handle_option(typename OptionBase<YCompCommandLine, T, self>::iterator it) const {
        // it points to the current argument
        // -> skip it
        ++it;
        do
        {
            std::string graph = *it++;
            bool temp = true;
            std::string file;

            std::string arg2;
            if(has_next(it)) {
                arg2 = *it++;

                if(arg2.compare("true") == 0) {
                    temp = true;
                    if(has_next(it)) {
                        file = *it++;
                    } else {
                        std::cerr << "Not enough args!" << std::endl;
                    }
                } else if(arg2.compare("false") == 0) {
                    temp = false;
                    if(has_next(it)) {
                        file = *it++;
                    }else {
                        std::cerr << "Not enough args!" << std::endl;
                    }
                } else {
                    file = arg2;
                }

                OptionBase<YCompCommandLine, T, self>::target()->add(graph, temp, file);
            } else {
                std::cerr << "Not enough args!" << std::endl;
            }
        } while (has_next(it));
        return it;
    }
};

template<>
struct Option<void, void> : public BasicOption<void>{
    Option()
        : BasicOption<void>("empty arg", "", nullptr)
    {}

    BasicOption::iterator handle(BasicOption::iterator it) const {
        std::cerr << "unknown argument: " << *it << std::endl;
        return ++it;
    }

    template<typename NData>
    Option<NData, Option<void, void>> add_option(const std::string& arg, const std::string& help_msg,
                                                 NData& target, const NData& def = NData()) const {
        return Option<NData, Option<void, void>>(*this, arg, help_msg, &target, def);
    }

    bool is_arg(const char* value) const { return true; }
    void print_help() {}
};

struct ImplicitOption : BasicOption<std::vector<std::string>> {
    typedef std::vector<std::string> target_type;

    ImplicitOption(const std::string& arg, const std::string& help_msg, target_type& target)
        : BasicOption<target_type>(arg, help_msg, &target)
    {}

    template<typename NData>
    Option<NData, ImplicitOption> add_option(const std::string& arg, const std::string& help_msg,
                                             NData& target, const NData& def = NData()) const {
        return Option<NData, ImplicitOption>(*this, arg, help_msg, &target, def);
    }

    BasicOption<target_type>::iterator handle(BasicOption<target_type>::iterator it) const {
        // append the data to the list
        target()->push_back(*it++);
        return it;
    }

    bool is_arg(const char* value) const { return true; }
    void print_help() { std::cout << "implicit arguments: " << help_msg() << std::endl; }
};

class ArgParser {
public:
    ArgParser() {}

    ImplicitOption implicit_option(const std::string& arg, const std::string& help_msg, std::vector<std::string>& target) const {
        return ImplicitOption(arg, help_msg, target);
    }

    Option<void, void> option() const { return Option<void, void>(); }
};

}

#endif
