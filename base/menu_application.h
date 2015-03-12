// File Description
// Author: Philip Salvaggio

#ifndef MENU_APPLICATION_H
#define MENU_APPLICATION_H

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <curses.h>
#include <menu.h>

namespace mats {

class MenuApplication {
 public:
  using action_t = std::function<void()>;
  class Menu;
  class MenuItem;

  MenuApplication();
  ~MenuApplication();

  void AddItem(const std::string& title,
               const std::string& description,
               const action_t& action);
  Menu* AddSubmenu(const std::string& title,
                   const std::string& description);
  void PostStatusMessage(const std::string& message);
  void run();
  void stop();

 private:
  void PushMenu(Menu* menu);
  void PopMenu();

 private:
  void FreeCursesMenus();
  void MakeCursesMenus();

 public:
  class MenuItem {
   public:
    MenuItem(const std::string& title,
             const std::string& description,
             const action_t& action);
    virtual ~MenuItem();
           
    const std::string& title() const { return title_; }
    void set_title(const std::string& title) { title_ = title; }

    const std::string& description() const { return description_; }
    void set_description(const std::string& description) {
      description_ = description;
    }

    void PerformAction() { action_(); }
    void set_action(const action_t& action) { action_ = action; }

    // Caller now owns the item.
    void GetCursesItem(ITEM** item);

    virtual bool is_submenu() const { return false; }

   private:
    std::string title_;
    std::string description_;
    action_t action_;
  };

  class Menu {
   public:
    Menu(MenuApplication* app, Menu* parent_ = nullptr);
    virtual ~Menu();
  
    void AddItem(const std::string& title, 
                 const std::string& description,
                 const action_t& action);
    void AddBackItem();

    Menu* AddSubmenu(const std::string& title,
                     const std::string& description);

    size_t size() const { return items_.size(); }

    // Caller owns the menu and the item list.
    void GetCursesMenu(MENU** menu, ITEM*** items);

    MenuApplication* app() { return app_; }
    Menu* parent() { return parent_; }
   private:
    std::vector<std::unique_ptr<MenuItem>> items_;
    MenuApplication* app_;
    Menu* parent_;
  };

  class Submenu : public MenuItem, public Menu {
   public:
    Submenu(const std::string& title,
            const std::string& description,
            MenuApplication* app,
            Menu* parent);

    bool is_submenu() const override { return true; }
  };

 private:
  bool keep_going_;
  Menu main_menu_;
  Menu* current_menu_;

  std::unique_ptr<ITEM*> current_curses_items_;
  MENU* current_curses_menu_;
};

}

#endif  // MENU_APPLICATION_H
