package com.happy.observator;

import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

import com.happy.observator.service.UserService;

@SpringBootApplication
public class ObservatorApplication {

	public static void main(String[] args) {
		SpringApplication.run(ObservatorApplication.class, args);
	}

	@Bean
    CommandLineRunner run(UserService userService) {
        return args -> {
            userService.saveUser("user", "password");
        };
    }
}
