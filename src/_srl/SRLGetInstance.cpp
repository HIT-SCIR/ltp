#include "GetInstance.h"
#include "Configuration.h"
#include <string>

using namespace std;

void print_usage(char* exe_path)
{
    string exe_name(exe_path);
    size_t find = exe_name.find_last_of("\\/");

    if (string::npos != find)
    {
        exe_name = exe_name.substr(find+1);
    }
    cout<<"This program get instance which is used for train or predict"<<endl
        <<"Usage:"<<endl
        <<"      "<<exe_name<<" [config.xml/IN] [feature folder/IN] [select.config/IN] [instances/OUT] [| isdevel]"<<endl;
}


int main(int argc, char* argv[])
{
    if (argc < 5)
    {
        print_usage(argv[0]);
        return 1;
    }
    string xml_file_name(argv[1]);
    Configuration configuration(xml_file_name);
    GetInstance get_instance(configuration);
    if (argc == 5)
    {
        get_instance.generate_argu_instance(
            argv[2],
            argv[3],
            argv[4]
        );
    }
    if (argc == 6)
    {
        get_instance.generate_argu_instance(
            argv[2],
            argv[3],
            argv[4],
            true
        );
    }

    return 0;
}
