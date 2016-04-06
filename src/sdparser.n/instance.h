#ifndef __LTP_PARSERN_INSTANCE_H__
#define __LTP_PARSERN_INSTANCE_H__

#include <iostream>
#include <vector>

namespace ltp {
namespace depparser {

class Instance {
private:
  typedef std::vector<int> node_t;
  typedef std::vector<node_t> tree_t;

public:
  Instance();
  ~Instance();

  size_t size() const;  //
  size_t num_heads(bool ignore_punctation = true) const;
  size_t num_recalled_heads(bool ignore_punctation = true) const;
  size_t num_recalled_deprels(bool ignore_punctation = true) const;
  bool is_tree() const;
  bool is_projective() const;
  bool is_non_projective() const;
public:
  std::vector<std::string> raw_forms; //! The original form.
  std::vector<std::string> forms;     //! The converted form.
  std::vector<std::string> lemmas;    //! The lemmas.
  std::vector<std::string> postags;   //! The postags.
  std::vector<std::string> cpostags;  //! The cpostags.

  std::vector<int>          heads;
  std::vector<int>          deprelsidx;
  std::vector<std::string > deprels;
  std::vector<int>          predict_heads;
  std::vector<int>          predict_deprelsidx;
  std::vector<std::string > predict_deprels;
private:
  bool is_tree(const std::vector<int>& heads) const;
  bool is_tree_travel(int now, const tree_t& tree, std::vector<bool>& visited) const;
  bool is_projective(const std::vector<int>& heads) const;
  bool is_non_projective(const std::vector<int>& heads) const;

};  // end for class Instance

struct Dependency {
  std::vector<int> forms;
  std::vector<int> postags;
  std::vector<int> heads;
  std::vector<int> deprels;

  size_t size() const;
};

} //  namespace depparser
} //  namespace ltp

#endif  //  end for __LTP_PARSERN_INSTANCE_H__
