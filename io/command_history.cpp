/**
 * File: CommandHistory.cpp
 *
 * Implementation file for the CommandHistory class.
 *
 * Author: Philip Salvaggio
 */

#include "command_history.h"

#include <iostream>

using namespace std;

namespace mats_io {

const string CommandHistory::kBlank("");


CommandHistory::CommandHistory(size_t capacity)
    : capacity_(capacity),
      history_(),
      current_cmd_(end(history_)) {}


CommandHistory::~CommandHistory() {}


void CommandHistory::AddCommand(const string& command) {
  history_.emplace_back(command);
  if (history_.size() > capacity_) history_.pop_front(); 
  ResetIterator();
}


const string& CommandHistory::operator*() const {
  if (current_cmd_ == end(history_)) return kBlank;

  return *current_cmd_;
}


void CommandHistory::operator++() {
  if (history_.empty() || current_cmd_ == begin(history_)) return;
  current_cmd_--;
}


void CommandHistory::operator--() {
  if (history_.empty() || current_cmd_ == end(history_)) return;
  current_cmd_++;
}


void CommandHistory::ResetIterator() {
  current_cmd_ = end(history_);
}

void CommandHistory::PrintHistory() const {
  cout << "Command History:" << endl;
  for (const auto& cmd : history_) cout << cmd << endl;
}

}
