//
// Created by liu on 2017/1/5.
//

#ifndef PROJECT_PROCESSLOADER_H
#define PROJECT_PROCESSLOADER_H

#include "config.h"
#include "debug.h"

namespace base{
  class ProcessLoader {
    int argc;
    char ** argv;
    Debug * debug;

  public:
    ProcessLoader(int argc, char * argv[]):argc(argc), argv(argv) { }

    template <class RunClass>
    int runProcess() {
      auto c = RunClass::createConfig();
      c.init(argc, argv);

      base::Debug::init(c);
      debug = new base::Debug("main");
      debug->debug(c.toString("\n", " -> "));

      RunClass processRunner(c);
      processRunner.main();
      return processRunner.returnNum;
    }
  };
}



#endif //PROJECT_PROCESSLOADER_H
