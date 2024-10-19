package com.happy.observator.repository;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;

import com.happy.observator.model.User;

public interface UserRepositary extends JpaRepository<User, Integer>{
    Optional<User> findByUsername(String username);
} 
