#ifndef STUDENT_H_
#define STUDENT_H_

#include <string>
#include <vector>

class Class;

enum Degree {
    undergraduate,
    graduate
};

class Student {
    // TODO: implement class Student.
private:
    const std::string year;
    const std::string name;
    Degree degree;
protected:
    std::vector<Class*> classes;
public:
    const std::string id;
    std::string toString() const;
    virtual double getGpa() const=0;
    virtual double getAvgScore() const=0;
    void addClass(Class* c);
    Student(std::string a,std::string b,std::string c):id(a),name(b),year(c){};
};

// TODO: implement class Graduate.
class Graduate:public Student{
public:
    double getGpa() const;
    double getAvgScore() const;
    Graduate(std::string a,std::string b,std::string c):Student(a,b,c){};
};
// TODO: implement class Undergraduate.
class Undergraduate:public Student{
public:
    double getGpa() const;
    double getAvgScore() const;
    Undergraduate(std::string a,std::string b,std::string c):Student(a,b,c){};
};
class StudentWrapper {
private:
    const Student &student;
    double score=0.0;
public:
    const std::string id;
    // TODO: fix error
    StudentWrapper(const std::string &id, const Student &student):id(id),student(student) {  }

    void setScore(double score)
    {
        if (score < 0 || score > 100)
            //throw "Invalid Score!";
        this->score = score;
    }

    double getScore() const
    {
        return this->score;
    }

    std::string toString() const
    {
        return student.toString();
    }
};

#endif // STUDENT_H_
