// Linux/X11 container for the original Pangolin renderer. No image forwarding.
#pragma once
#include <pangolin/pangolin.h>
#include <pangolin/display/device/X11Window.h>
#include <stdexcept>
#include <string>

namespace ORB_SLAM3 {
class EmbeddedViewerWindow {
public:
    EmbeddedViewerWindow(const std::string &name, unsigned long parent) : name_(name), parent_(parent) {
        auto &window = pangolin::CreateWindowAndBind(name_, 1000, 400, pangolin::Params{{"scheme", "x11"}});
        window_ = dynamic_cast<pangolin::X11Window*>(&window);
        if(!window_) {
            pangolin::DestroyWindow(name_);
            throw std::runtime_error("Embedded ORB preview requires Pangolin's X11 backend");
        }
        try {
            auto *display = window_->display->display;
            // Pangolin initially maps a managed top-level window. Withdraw it
            // and opt out of WM management before making it an embedded child.
            // Unmapping alone lets a delayed WM request detach it again.
            XWithdrawWindow(display, window_->win, DefaultScreen(display));
            XSetWindowAttributes attributes{};
            attributes.override_redirect = True;
            XChangeWindowAttributes(display, window_->win, CWOverrideRedirect, &attributes);
            Resize();
            XSync(display, False);
        } catch(...) {
            pangolin::DestroyWindow(name_);
            throw;
        }
        // Embedded children must not detach from their host via fullscreen hotkeys.
        pangolin::RegisterKeyPressCallback('\t', []{});
        pangolin::RegisterKeyPressCallback(pangolin::PANGO_KEY_ESCAPE, []{});
    }
    ~EmbeddedViewerWindow() { pangolin::DestroyWindow(name_); }
    void Resize() {
        auto *display = window_->display->display;
        XWindowAttributes attributes;
        if(!XGetWindowAttributes(display, parent_, &attributes))
            throw std::runtime_error("ORB preview parent window is unavailable");
        // WM unmanage/reparent events can arrive after the initial XSync.
        // Check throughout rendering, including while a map reset is paused.
        Window root, actualParent, *children = nullptr;
        unsigned int count = 0;
        const bool queried = XQueryTree(display, window_->win, &root, &actualParent, &children, &count);
        if(children) XFree(children);
        if(!queried) throw std::runtime_error("Cannot query the ORB preview window");
        const bool detached = actualParent != parent_;
        if(detached) {
            XReparentWindow(display, window_->win, parent_, 0, 0);
            XMapWindow(display, window_->win);
        }
        if(detached || attributes.width != width_ || attributes.height != height_) {
            width_ = std::max(1, attributes.width);
            height_ = std::max(1, attributes.height);
            window_->Resize(width_, height_);
        }
        // Delayed ConfigureNotify events may reset Pangolin's viewport to the
        // former top-level size even though the embedded host has not changed.
        const auto &viewport = pangolin::DisplayBase().v;
        if(viewport.w != width_ || viewport.h != height_)
            pangolin::process::Resize(width_, height_);
    }
private:
    std::string name_;
    unsigned long parent_;
    pangolin::X11Window *window_ = nullptr;
    int width_ = 0, height_ = 0;
};
}
