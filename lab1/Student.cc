#include "Student.h"
#include <string>
#include <sstream>
#include "Class.h"

std::string Student::toString() const
{
    // TODO: uncomment the following code after implementing class Student.

    std::ostringstream oss;
    oss << "Student Information:"
        << "\n\tid: " << id
        << "\n\tname: " << name
        << "\n\tenrollment year: " << year
        << "\n\tdegree: " << (degree == graduate ? "graduate" : "undergraduate")
        << std::endl;
    return oss.str();

}

// TODO: implement functions which are declared in Student.h.
double Undergraduate::getAvgScore() const {
    double temp=0.0;
    int sum=0;
    for(int i=0;i<=classes.size()-1;i++){
        sum+=classes[i]->point;
        temp+=classes[i]->point*classes[i]->getStudentWrapper(id).getScore();
    }
    if(sum!=0)return temp/sum;
    else return 0.00;
}
double Undergraduate::getGpa() const {
    double temp=0.0;
    int sum=0;
    for(int i=0;i<=classes.size()-1;i++){
        sum+=classes[i]->point;
        temp+=(classes[i]->point*classes[i]->getStudentWrapper(id).getScore())/20;
    }
    if(sum!=0)return temp/sum;
    else return 0.00;
}
double Graduate::getAvgScore() const {
    double temp=0.0;
    int sum=0;
    for(int i=0;i<=classes.size()-1;i++){
        sum+=classes[i]->point;
        temp+=classes[i]->point*classes[i]->getStudentWrapper(id).getScore();
    }
    if(sum!=0)return temp/sum;
    else return 0.00;
}
double Graduate::getGpa() const {
    double temp=0.0,tempGpa;
    int tempscore,sum=0;
    for(int i=0;i<=classes.size()-1;i++){
        sum+=classes[i]->point;
        tempscore=classes[i]->getStudentWrapper(id).getScore();
        if(tempscore>=90 && tempscore<=100)tempGpa=4.00;
        else if(tempscore>=80 && tempscore<=89)tempGpa=3.50;
        else if(tempscore>=70 && tempscore<=79)tempGpa=3.00;
        else if(tempscore>=60 && tempscore<=69)tempGpa=2.50;
        else if(tempscore<60)tempGpa=2.00;
        temp+=classes[i]->point*tempGpa;
    }
    if(sum!=0)return temp/sum;
    else return 2.00;
}
void Student::addClass(Class* cl){
    this->classes.push_back(cl);
}