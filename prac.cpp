#include <iostream>
#include <vector>
#include <string>

int main(){
    int x = 121;
    std::string palindrome_check = std::to_string(x);
    std::string check;
    for(int i = palindrome_check.length(); i >= 0; i--){
        check.append(palindrome_check.substr(i, 1));
    }

    std::cout << check;
    return 0;
}

