package com.happy.observator.model;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.validation.constraints.NotBlank;

@Entity
@Table(name = "users")
public class User {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private int id;

    @NotBlank
    private String username;

    @NotBlank
    private String password;

    private boolean isCertified;
    private long balance;


    public int getId() {
        return this.id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getUsername() {
        return this.username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getPassword() {
        return this.password;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    public boolean isIsCertified() {
        return this.isCertified;
    }

    public boolean getIsCertified() {
        return this.isCertified;
    }

    public void setIsCertified(boolean isCertified) {
        this.isCertified = isCertified;
    }

    public long getBalance() {
        return this.balance;
    }

    public void setBalance(long balance) {
        this.balance = balance;
    }


    public User() {}
    
    public User(String username, String password) {
        this.username = username;
        this.password = password;
    }
}
