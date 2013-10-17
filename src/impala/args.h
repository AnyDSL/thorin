#include <iterator>
#include <vector>
#include <hash_map>

#ifndef IMPALA_ARGS_H_
#define IMPALA_ARGS_H_

class ArgsIterator : public std::iterator<std::forward_iterator_tag, const char*> {
public:
    ArgsIterator(int pos, char** argv)
        : pos_(pos)
        , argv_(argv)
    { }

    const char* operator*() const { return argv_[pos_]; }

    ArgsIterator& operator++() {
        ++pos_;
        return *this;
    }

    ArgsIterator operator++(int) {
        ArgsIterator tmp(pos_, argv_);
        ++pos_;
        return tmp;
    }

    bool operator==(ArgsIterator b) const {
        return pos_ == b.pos_;
    }

    bool operator!=(ArgsIterator b) const {
        return pos_ != b.pos_;
    }

private:
    int pos_;
    char** argv_;
};

template<typename T>
class BasicOption {
public:
    BasicOption(const std::string& arg, const std::string& help_msg, T* target)
        : target_(target)
        , arg_(arg)
        , help_msg_(help_msg)
    { }

    T* target() const { return target_; }
    const std::string& arg() const { return arg_; }
    const std::string& help_msg() const { return help_msg_; }

    typedef ArgsIterator iterator;

    // methods

    //iterator handle(iterator it) const;

private:
    T* target_;
    std::string arg_;
    std::string help_msg_;
};

template<typename T, typename Data>
struct Option;

template<typename T, typename Data, typename Class>
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
        return strcmp(arg().c_str(), value + 1) == 0;
    }

    BasicOption<Data>::iterator handle(BasicOption<Data>::iterator it) const {
        const Class* c = as_class();
        if (!*it)
            return it;
        if (c->is_arg(*it))
            return c->handle_option(it);
        return previous_.handle(it);
    }

    const Class* as_class() const { return static_cast<const Class*>(this); }

    template<typename NData>
    Option<Class, NData> add_option(const std::string& arg, const std::string& help_msg, NData& target, const NData& def = NData()) const {
        return Option<Class, NData>(*as_class(), arg, help_msg, &target, def);
    }

    void parse(int argc, char** argv) const {
        // parse command line arguments
        typedef BasicOption<Data>::iterator iter;
        for (iter it(1, argv), e(argc, argv); it != e;) {
            auto new_it = handle(it);
            assert(it != new_it);
            it = new_it;
        }
    }

    void print_help() {
        // print some help infos
        // TODO
    }

protected:
    T previous_;
};

template<typename T>
struct EndOption {
    EndOption(const T& previous)
        : previous_(previous)
    { }


private:
    T previous_;
};

template<typename T, typename Data>
struct Option : public OptionBase<T, Data, Option<T, Data>> {
    Option(const T& previous, const std::string& arg, const std::string& help_msg, Data* target, const Data& def = Data())
        : OptionBase<T, Data, Option<T, Data>>(previous, arg, help_msg, target, def)
    { }

    typename OptionBase<T, Data, Option<T, Data>>::iterator handle_option(typename OptionBase<T, Data, Option<T, Data>>::iterator it) const {
        static_assert(false, "illegal use of program option");
    }
};

template<typename T>
struct Option<T, int> : public OptionBase<T, int, Option<T, int>> {
    typedef Option<T, int> self;

    Option(const T& previous, const std::string& arg, const std::string& help_msg, int* target, int def)
        : OptionBase<T, int, Option<T, int>>(previous, arg, help_msg, target, def)
    { }

    typename OptionBase<T, int, self>::iterator handle_option(typename OptionBase<T, int, self>::iterator it) const {
        *target() = atoi(*++it);
        return ++it;
    }
};

template<typename T>
struct Option<T, bool> : public OptionBase<T, bool, Option<T, bool>> {
    typedef Option<T, bool> self;

    Option(const T& previous, const std::string& arg, const std::string& help_msg, bool* target, bool def)
        : OptionBase<T, bool, self>(previous, arg, help_msg, target, def)
    { }

    typename OptionBase<T, bool, self>::iterator handle_option(typename OptionBase<T, bool, self>::iterator it) const {
        *target() = true;
        return ++it;
    }

};

template<typename T>
struct Option<T, std::string> : public OptionBase<T, std::string, Option<T, std::string>> {
    typedef Option<T, std::string> self;

    Option(const T& previous, const std::string& arg, const std::string& help_msg, std::string* target, const std::string& def)
        : OptionBase<T, std::string, self>(previous, arg, help_msg, target, def)
    { }

    typename OptionBase<T, std::string, self>::iterator handle_option(typename OptionBase<T, std::string, self>::iterator it) const {
        *target() = *++it;
        return ++it;
    }
};

typedef std::vector<std::string> OptionStringVector;

template<typename T>
struct Option<T, OptionStringVector> : public OptionBase<T, OptionStringVector, Option<T, OptionStringVector>> {
    typedef Option<T, OptionStringVector> self;

    Option(const T& previous, const std::string& arg, const std::string& help_msg, OptionStringVector* target, const OptionStringVector& def)
        : OptionBase<T, OptionStringVector, self>(previous, arg, help_msg, target, def)
    { }

    typename OptionBase<T, OptionStringVector, self>::iterator handle_option(typename OptionBase<T, OptionStringVector, self>::iterator it) const {
        do
        {
            target()->push_back(*it++);
        } while (*it && (*it)[0] != '-');
        return it;
    }
};


template<>
struct Option<void, void> : public BasicOption<void>{
    Option()
        : BasicOption<void>("empty arg", "", nullptr)
    { }

    bool is_arg(const char* value) const { return true; }

    BasicOption::iterator handle(BasicOption::iterator it) const {
        std::cerr << "unknown argument: " << *it << std::endl;
        return ++it;
    }

    template<typename NData>
    Option<Option<void, void>, NData> add_option(const std::string& arg, const std::string& help_msg,
                                                 NData& target, const NData& def = NData()) const {
        return Option<Option<void, void>, NData>(*this, arg, help_msg, &target, def);
    }
};

struct ImplicitOption : BasicOption<std::vector<std::string>> {
    typedef std::vector<std::string> target_type;

    ImplicitOption(const std::string& arg, const std::string& help_msg, target_type& target)
        : BasicOption<target_type>(arg, help_msg, &target)
    { }

    template<typename NData>
    Option<ImplicitOption, NData> add_option(const std::string& arg, const std::string& help_msg,
                                             NData& target, const NData& def = NData()) const {
        return Option<ImplicitOption, NData>(*this, arg, help_msg, &target, def);
    }

    bool is_arg(const char* value) const { return true; }

    BasicOption<target_type>::iterator handle(BasicOption<target_type>::iterator it) const {
        // append the data to the list
        target()->push_back(*it++);
        return it;
    }
};

class ArgParser {
public:
    ArgParser(const std::string& program_desc)
        : program_desc_(program_desc)
    { }

    ImplicitOption implicit_option(const std::string& arg, const std::string& help_msg, std::vector<std::string>& target) const {
        return ImplicitOption(arg, help_msg, target);
    }

    Option<void, void> option() const {
        return Option<void, void>();
    }

    const std::string& program_desc() const { return program_desc_; }

private:
    std::string program_desc_;
};

#endif
