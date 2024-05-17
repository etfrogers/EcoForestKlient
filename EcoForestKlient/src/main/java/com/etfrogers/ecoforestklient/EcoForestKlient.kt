package com.etfrogers.ecoforestklient

class EcoForestKlient {
    fun getString() : String{
        return "Hello from EcoForest!"
    }
}

fun main(){
    println(EcoForestKlient().getString())
}