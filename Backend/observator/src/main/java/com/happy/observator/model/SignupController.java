package com.happy.observator.model;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;

import com.happy.observator.service.UserService;

@Controller
public class SignupController {
    
    private UserService userService;

    public SignupController(UserService userService) {
        this.userService = userService;
    }
    
    @GetMapping("/signup")
    public String showSignupForm(Model model) {
        model.addAttribute("user", new User());
        return "signup";
    }

    @PostMapping("/signup")
    public String registerUser(@ModelAttribute User user, Model model) {
        try{
            userService.saveUser(user.getUsername(), user.getPassword());
            return "redirect:/login?signup=true";  // Redirect to login with signup success message
        } catch(IllegalArgumentException e){
            model.addAttribute("error", e.getMessage());
            return "signup";
        }
        
    }
}