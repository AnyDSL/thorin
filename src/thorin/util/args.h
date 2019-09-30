#ifndef THORIN_UTILS_H
#define THORIN_UTILS_H

#include <cassert>
#include <iomanip>
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

    BasicOption(std::string param, std::string domain, std::string help, T* target)
        : target_(target)
        , param_(param)
        , domain_(domain)
        , help_(help)
    {}
    virtual ~BasicOption() {}

    T* target() const { return target_; }
    const std::string& param() const { return param_; }
    const std::string& domain() const { return domain_; }
    const std::string& help() const { return help_; }

protected:
    T* target_;
    std::string param_;
    std::string domain_;
    std::string help_;
};

template<typename T, typename Data>
struct Option;

template<typename Data, typename T, typename Class>
class OptionBase : public BasicOption<Data> {
public:
    OptionBase(const T& previous, std::string param, std::string domain, std::string help, Data* target, const Data& init)
        : BasicOption<Data>(param, domain, help, target)
        , previous_(previous)
    {
        *target = init;
    }

    bool is_param(const char* value) const {
        // TODO: nicer
        if (value[0] != '-')
            return false;

        // allow also --
        int offset = 1;
        if (value[1] == '-') offset++;

        return strcmp(BasicOption<Data>::param().c_str(), value + offset) == 0;
    }

    typename BasicOption<Data>::iterator handle(typename BasicOption<Data>::iterator it) const {
        const Class* c = as_class();
        if (!*it)
            return it;
        if (c->is_param(*it))
            return c->handle_option(it);
        return previous_.handle(it);
    }

    const Class* as_class() const { return static_cast<const Class*>(this); }

    template<typename NData>
    Option<NData, Class> add_option(std::string param, std::string domain, std::string help, NData& target, const NData& init = {}) const {
        return Option<NData, Class>(*as_class(), param, domain, help, &target, init);
    }

    void parse(int argc, char** argv) const {
        typedef typename BasicOption<Data>::iterator iter;
        for (iter it(1, argv), e(argc, argv); it != e;) {
            auto new_it = handle(it);
            assert(it != new_it);
            it = new_it;
        }
    }

    void print_help() {
        const Class* c = as_class();
        previous_.print_help();
        std::cout << "  " << '-' << std::left << std::setw(36) << (c->param() + std::string(" ") + c->domain()) << c->help() << std::endl;
    }

protected:
    T previous_;
};

template<typename Data, typename T>
struct Option : public OptionBase<Data, T, Option<Data, T>> {
    Option(const T& previous, std::string param, std::string domain, std::string help, Data* target, const Data& init = Data())
        : OptionBase<Data, T, Option<T, Data>>(previous, param, domain, help, target, init)
    {}

    typename OptionBase<Data, T, Option<T, Data>>::iterator handle_option(typename OptionBase<Data, T, Option<T, Data>>::iterator) const {
        assert(false && "illegal use of program option");
    }
};

template<typename T>
struct Option<int, T> : public OptionBase<int, T, Option<int, T>> {
    typedef Option<int, T> self;

    Option(const T& previous, std::string param, std::string domain, std::string help, int* target, int init)
        : OptionBase<int, T, self>(previous, param, domain, help, target, init)
    {}

    typename OptionBase<int, T, self>::iterator handle_option(typename OptionBase<int, T, self>::iterator it) const {
        *OptionBase<int, T, self>::target() = atoi(*++it);
        return ++it;
    }
};

template<typename T>
struct Option<bool, T> : public OptionBase<bool, T, Option<bool, T>> {
    typedef Option<bool, T> self;

    Option(const T& previous, std::string param, std::string domain, std::string help, bool* target, bool init)
        : OptionBase<bool, T, self>(previous, param, domain, help, target, init)
    {}

    typename OptionBase<bool, T, self>::iterator handle_option(typename OptionBase<bool, T, self>::iterator it) const {
        *OptionBase<bool, T, self>::target() = true;
        return ++it;
    }

};

template<typename T>
struct Option<std::string, T> : public OptionBase<std::string, T, Option<std::string, T>> {
    typedef Option<std::string, T> self;

    Option(const T& previous, std::string param, std::string domain, std::string help, std::string* target, std::string init)
        : OptionBase<std::string, T, self>(previous, param, domain, help, target, init)
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

    Option(const T& previous, std::string param, std::string domain, std::string help, OptionStringVector* target, const OptionStringVector& init)
        : OptionBase<OptionStringVector, T, self>(previous, param, domain, help, target, init)
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

template<>
struct Option<void, void> : public BasicOption<void> {
    Option()
        : BasicOption<void>("empty arg", "", "", nullptr)
    {}

    BasicOption::iterator handle(BasicOption::iterator it) const {
        std::cerr << "unknown argument: " << *it << std::endl;
        return ++it;
    }

    template<typename NData>
    Option<NData, Option<void, void>> add_option(std::string param, std::string domain, std::string help, NData& target, const NData& init = {}) const {
        return Option<NData, Option<void, void>>(*this, param, domain, help, &target, init);
    }

    bool is_param(const char*) const { return true; }
    void print_help() {}
};

struct ImplicitOption : BasicOption<std::vector<std::string>> {
    typedef std::vector<std::string> target_type;

    ImplicitOption(std::string domain, std::string help, target_type& target)
        : BasicOption<target_type>("", domain, help, &target)
    {}

    template<typename NData>
    Option<NData, ImplicitOption> add_option(std::string param, std::string domain, std::string help, NData& target, const NData& init = {}) const {
        return Option<NData, ImplicitOption>(*this, param, domain, help, &target, init); }

    BasicOption<target_type>::iterator handle(BasicOption<target_type>::iterator it) const {
        target()->push_back(*it++);
        return it;
    }

    bool is_param(const char*) const { return true; }
    void print_help() { std::cout << "implicit arguments: " << help() << std::endl; }
};

class ArgParser {
public:
    ArgParser() {}

    ImplicitOption implicit_option(std::string domain, std::string help, std::vector<std::string>& target) const {
        return ImplicitOption(domain, help, target);
    }

    Option<void, void> option() const { return Option<void, void>(); }
};

}

#endif
