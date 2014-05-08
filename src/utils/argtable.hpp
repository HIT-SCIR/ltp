#ifndef __LTP_UTILS_ARGTABLE_HPP__
#define __LTP_UTILS_ARGTABLE_HPP__

#include <iostream>
#include <cstring>
#include <vector>

namespace ltp {
namespace utility {

class OptionParser {
public:
    enum {
        ARG_LIT = 0,
        ARG_INT,
        ARG_DBL,
        ARG_STR};

    /**
     * The constructor
     */
    OptionParser() : usage(0) {
    }

    OptionParser(const char * _usage) {
        if (_usage) {
            int len = strlen(_usage);
            usage = new char[len];
            strcpy(usage, _usage);
        }
    }

    /**
     * The destructor
     */
    ~OptionParser() {
        for (int i = 0; i < entries.size(); ++ i) {
            delete entries[i];
        }

        if (usage) delete [](usage);
    }

    /**
     * Add an option.
     *
     *  @param[in]  short_opt   short option name for the option.
     *  @param[in]  long_opt    long option name for the option.
     *  @param[in]  data_type   data type for the option.
     *  @param[in]  dest_name   destination name for the option.
     *  @param[in]  glossary    the glossary for the option.
     */
    void add_option(const char *short_opt,
            const char *long_opt,
            const char *data_type,
            const char *dest_name,
            const char *glossary) {
        arg_entry * entry = new arg_entry();

        int len;
        len = strlen(short_opt); entry->short_opt = new char[len + 1]; strcpy(entry->short_opt, short_opt);
        len = strlen(long_opt);  entry->long_opt = new char[len + 1];  strcpy(entry->long_opt, long_opt);
        len = strlen(dest_name); entry->dest_name = new char[len + 1]; strcpy(entry->dest_name, dest_name);
        len = strlen(glossary);  entry->glossary = new char[len + 1];  strcpy(entry->glossary, glossary);

        if (strcmp(data_type, "lit") == 0 || strcmp(data_type, "literal") == 0) {
            entry->data_type = ARG_LIT;
            entry->value = (void *)(new char);
        } else if (strcmp(data_type, "int") == 0 || strcmp(data_type, "integer") == 0) {
            entry->data_type = ARG_INT;
            entry->value = (void *)(new int);
        } else if (strcmp(data_type, "dbl") == 0 || strcmp(data_type, "double") == 0) {
            entry->data_type = ARG_DBL;
            entry->value = (void *)(new double);
        } else if (strcmp(data_type, "str") == 0 || strcmp(data_type, "string") == 0) {
            entry->data_type = ARG_STR;
            entry->value = (void *)(new const char *);
        }

        entries.push_back( entry );
    }

    /**
     * Parse the option from command line argument.
     *
     *  @param[in]  argc    number of arguments.
     *  @param[in]  argv    string for arguments.
     */
    int parse_args(int argc, const char **argv) {
        for (int i = 0; i < argc; ++ i) {
            int len = strlen(argv[i]);
            int entry_idx = -1;

            if (argv[i][0] == '-') {
                if (len > 1 && argv[i][1] == '-') { // long option identification
                    for (entry_idx = 0;
                            entry_idx < entries.size() && strcmp(entries[entry_idx]->long_opt, argv[i] + 2);
                            ++ entry_idx);
                } else { // short option identification
                    for (entry_idx = 0;
                            entry_idx < entries.size() && strcmp(entries[entry_idx]->short_opt, argv[i] + 1);
                            ++ entry_idx);
                }

                // not found.
                if (entry_idx == entries.size()) {
                    return -1;
                } else {
                    switch(entries[entry_idx]->data_type) {
                        case ARG_LIT:
                            arg_set_entry( entries[entry_idx], "" );
                            break;
                        case ARG_INT:
                            arg_set_entry( entries[entry_idx], argv[++ i]);
                            break;
                        case ARG_DBL:
                            arg_set_entry( entries[entry_idx], argv[++ i]);
                            break;
                        case ARG_STR:
                            arg_set_entry( entries[entry_idx], argv[++ i]);
                            break;
                        default:
                            return -1;
                    }
                }
            } else {
                return -1;
            }
        }
        return 0;
    }

    /**
     * Print the glossary for the option.
     */
    void print_glossary() {
        if (usage != NULL) {
            fprintf(stderr, "usage: %s\n\n", usage);
        } else {
            fprintf(stderr, "usage: ./excuatable [option]\n\n");
        }

        for (int i = 0; i < entries.size(); ++ i) {
            fprintf(stderr, "\t-%s --%-19s %s\n",
                    entries[i]->short_opt,
                    entries[i]->long_opt,
                    entries[i]->glossary);
        }

        fprintf(stderr, "\n");
    }

    /**
     * Return the pointer to the value
     *
     *  @param[in]  dest_name   destination name for the option.
     *  @return pointer to the value,
     *          NULL on fail.
     */
    void *option(const char *dest_name) {
        arg_entry *entry = arg_find_entry(dest_name);
        if (entry == NULL) {
            return NULL;
        } else {
            return entry->value;
        }
    }

private:
    class arg_entry {
    public:
        arg_entry() : value(NULL) {}
        ~arg_entry() {
            switch (data_type) {
                case ARG_LIT:
                    delete (char *)value;
                    break;
                case ARG_INT:
                    delete (int *)value;
                    break;
                case ARG_DBL:
                    delete (double *)value;
                    break;
                case ARG_STR:
                    delete (const char **)value;
                    break;
                default:
                    break;
            }

            if (short_opt)  delete [](short_opt);
            if (long_opt)   delete [](long_opt);
            if (dest_name)  delete [](dest_name);
            if (glossary)   delete [](glossary);
        }

        char *short_opt;
        char *long_opt;
        char *dest_name;
        char *glossary;
        int data_type;
        void *value;
    };

private:
    char * usage;
    std::vector<arg_entry *> entries;

private:
    void arg_set_entry(arg_entry *entry, const char *value) {
        if (entry->data_type == ARG_LIT) {
            *((bool *)entry->value) = true;
        } else if (entry->data_type == ARG_INT) {
            *((int *)entry->value) = atoi(value);
        } else if (entry->data_type == ARG_DBL) {
            *((double *)entry->value) = atof(value);
        } else if (entry->data_type == ARG_STR) {
            *((const char **)entry->value) = value;
        }
    }

    arg_entry *arg_find_entry(const char *dest_name) {
        for (int i = 0; i < entries.size(); ++ i) {
            if (strcmp(entries[i]->dest_name, dest_name) == 0) {
                return entries[i];
            }
        }
        return NULL;
    }
};

}       //  end for namespace utility
}       //  end for namespace ltp
#endif  //  end for __LTP_UTILS_ARGTABLE_HPP__
