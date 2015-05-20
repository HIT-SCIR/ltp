#ifndef __LTP_PARSERN_TRANSITION_SYSTEM_H__
#define __LTP_PARSERN_TRANSITION_SYSTEM_H__

#include <iostream>
#include <vector>
#include "parser.n/instance.h"

namespace ltp {
namespace depparser {

class AbstractInexactAction {
protected:
  size_t seed;
public:
  AbstractInexactAction(): seed(0) {}

  /**
   * Constructor for inexact action. Empirically, the number of name
   * is less than 32. So such inexact action type compile the action
   * name and action type into a single integer.
   *
   *  @param[in]  name  The name for the action.
   *  @param[in]  rel   The dependency relation.
   */
  AbstractInexactAction(int name, int rel): seed(rel << 6 | name) {}

  bool operator == (const AbstractInexactAction& a) const { return (a.seed == seed); }
  bool operator != (const AbstractInexactAction& a) const { return (a.seed != seed); }
  bool operator <  (const AbstractInexactAction& a) const { return (seed < a.seed);  }

  inline int name() const { return (seed & 0x3f);   }
  inline int rel()  const { return (seed >> 6); }
};

class Action: public AbstractInexactAction {
public:
  enum {
    kNone = 0,  //! Placeholder for illegal action.
    kShift,     //! The index of shift action.
    kLeftArc,   //! The index of arc left action.
    kRightArc   //! The index of arc right action.
  };

  Action(): AbstractInexactAction() {}

  /**
   * Constructor for action.
   *
   *  @param[in]  name  The name for the action.
   *  @param[in]  rel   The dependency relation.
   */
  Action(int name, int rel): AbstractInexactAction(name, rel) {}

  //! Overload the ostream function.
  friend std::ostream& operator<<(std::ostream& os, const Action& act);

  //! For is_shift, is_left_arc, is_right_arc;
  friend class ActionUtils;
};

class ActionFactory {
public:
  /**
   * Make a shift action.
   *
   *  @return Action  A shift action.
   */
  static Action make_shift();

  /**
   * Make a arc left action.
   *
   *  @param[in]  rel     The dependency relation.
   *  @return     Action  The arc left action.
   */
  static Action make_left_arc(const int& rel);

  /**
   * Make a arc right action.
   *
   *  @param[in]  rel     The dependency relation.
   *  @return     Action  The arc right action.
   */
  static Action make_right_arc(const int& rel);
};

class ActionUtils {
public:
  /**
   * Calculate the orcale action sequence in the arcstandard transition system.
   *
   *  @param[in]  instance  The input reference instance.
   *  @param[out] actions   The oracle transition sequence.
   */
  static void get_oracle_actions(const Dependency& instance,
      std::vector<Action>& actions);

  static void get_oracle_actions2(const Dependency& instance,
      std::vector<Action>& actions);

  static void get_oracle_actions(const std::vector<int>& heads,
      const std::vector<int>& deprels,
      std::vector<Action>& actions);

  static void get_oracle_actions2(const std::vector<int>& heads,
      const std::vector<int>& deprels,
      std::vector<Action>& actions);

  /**
   * Judge if the input action is a shift action.
   *
   *  @param[in]  act   The action.
   *  @return     bool  Return true on the action being a shift action, otherwise
   *                    false.
   */
  static bool is_shift(const Action& act);

  /**
   * Judge if the input action is a RightArc action.
   *
   *  @param[in]  act     The action.
   *  @param[out] deprel  The dependency relation. If input is not ArcLeft, set it
   *                      to zero.
   *  @return     bool    Return true on the action being a shift action,
   *                      otherwise false.
   */
  static bool is_right_arc(const Action& act, int& deprel);

  /**
   * Judge if the input action is a LeftArc action.
   *
   *  @param[in]  act     The action.
   *  @param[out] deprel  The dependency relation. If input is not ArcLeft, set it
   *                      to zero.
   *  @return     bool    Return true on the action being a shift action,
   *                      otherwise false.
   */
  static bool is_left_arc(const Action& act, int& deprel);
private:
  //! The tree type.
  typedef std::vector<std::vector<int> > tree_t;

  /**
   * Perform the mid-order tree travel to get the correct actions sequence.
   *
   *  @param[in]  root      The current root to visit.
   *  @param[in]  instance  The reference instance.
   *  @param[in]  tree      The converted tree.
   *  @param[out] actions   The actions.
   */
  static void get_oracle_actions_travel(int root,
      const std::vector<int>& heads,
      const std::vector<int>& deprels,
      const tree_t& tree,
      std::vector<Action>& actions);

  static void get_oracle_actions_onestep(
      const std::vector<int>& heads,
      const std::vector<int>& deprels,
      std::vector<int>& sigma,
      int& beta,
      std::vector<int>& output,
      std::vector<Action>& actions);
};

class State {
public:
  State();  //! The empty constructor.

  /**
   * The constructor for the State.
   *
   *  @param[in]  r   The pointer to the dependency state.
   */
  State(const Dependency* r);

  /**
   * This method is needed by @class TransitionSystem.
   *
   *  @param[in]  source  The source of state to copy from.
   */
  void copy(const State& source);

  //! Clear the state.
  void clear();

  /**
   * Perform the shift action from source state.
   *
   *  @param[in]  source  The source state.
   */
  bool shift(const State& source);

  /**
   * Perform the left arc action from source state onto current state.
   *
   *  @param[in]  source  The source state.
   *  @param[in]  deprel  The dependency relation.
   */
  bool left_arc(const State& source, int deprel);

  /**
   * Perform the right arc action from source state onto current state.
   *
   *  @param[in]  source  The source state.
   *  @param[in]  deprel  The dependency relation.
   */
  bool right_arc(const State& source, int deprel);

  //! Used in dynamic oracle, should only be performed on gold state.
  int cost(const std::vector<int>& heads, const std::vector<int>& deprels);

  //! Return true on the buffer is empty.
  bool buffer_empty() const;

  //! Get the size of the stack.
  size_t stack_size() const;

  //! The pointer to the previous state.
  std::vector<int> stack;

  int buffer;               //! The front word in the buffer.
  const State* previous;    //! The pointer to the previous state.
  const Dependency* ref;    //! The pointer to the dependency tree.
  double score;             //! The score.
  Action last_action;       //! The last action.

  int top0;                 //! The top word on the stack.
  int top1;                 //! The second top word on the stack.
  std::vector<int> heads;   //! Use to record the heads in current state.
  std::vector<int> deprels; //! The dependency relation cached in state.
  std::vector<int> nr_left_children;      //! The number of left children in this state.
  std::vector<int> nr_right_children;     //! The number of right children in this state.
  std::vector<int> left_most_child;       //! The left most child for each word in this state.
  std::vector<int> right_most_child;      //! The right most child for each word in this state.
  std::vector<int> left_2nd_most_child;   //! The left 2nd-most child for each word in this state.
  std::vector<int> right_2nd_most_child;  //! The right 2nd-most child for each word in this state.

private:
  void refresh_stack_information();   //! Refresh the value of top0 and top1.
  bool can_shift() const;             //! Return can perform shift action.
  bool can_left_arc() const;          //! Return can perform left arc action.
  bool can_right_arc() const;         //! Return can perform right arc action.
};


class TransitionSystem {
private:
  size_t L;
  int R;
  int D;
public:
  TransitionSystem();  //! Constructor

  void set_dummy_relation(int d);
  void set_root_relation(int r);
  void set_number_of_relations(size_t l);
  //
  void get_possible_actions(const State& source, std::vector<Action>& actions);
  //
  void transit(const State& source, const Action& act, State* target);

  //
  std::vector<int> transform(const std::vector<Action>& actions);

  //
  void transform(const std::vector<Action>& actions, std::vector<int>& classes);

  int transform(const Action& act);

  Action transform(int act);

  //
  size_t number_of_transitions() const;
};

} //  end for namespace depparser
} //  end for namespace ltp

#endif  //  end for __LTP_PARSERN_TRANSITION_SYSTEM_H__
