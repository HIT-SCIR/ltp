#ifndef __LTP_SRL_OPTIONS_H__
#define __LTP_SRL_OPTIONS_H__

// namespace ltp {
// namespace srl {

struct TrainOptions {
    std::string     prg_train_file;
    std::string     srl_train_file;
    std::string     core_config; // Chinese.xml
    std::string     srl_config;  // srl.cfg
    std::string     srl_feature_dir;
    std::string     srl_instance_file;
    std::string     prg_instance_file;
    std::string     srl_model_file;
    std::string     prg_model_file;
    std::string     dst_config_dir; // destination cfgs
};

struct TestOptions {
    std::string     test_file;
    std::string     config_dir;
    std::string     output_file;
};

#endif
