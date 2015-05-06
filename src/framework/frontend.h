#ifndef __LTP_FRAMEWORK_FRONTEND_H__
#define __LTP_FRAMEWORK_FRONTEND_H__

namespace ltp {
namespace framework {

enum FrontendMode {
  kLearn,
  kTest,
  kDump
};

class Frontend {
protected:
  FrontendMode mode;

public:
  Frontend(const FrontendMode& _mode): mode(_mode) {}
};

}   //  namespace framework
}   //  namespace framework

#endif  //  end for __LTP_FRAMEWORK_FRONTEND_H__
