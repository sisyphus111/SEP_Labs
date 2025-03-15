#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "Class.h"
#include "Student.h"

using namespace std;

class AppX {
private:
    vector<Student *> studentVec;
    vector<Class *> classVec;

    void loadFiles();
    void inputScore();
    void printHighestScore();
    void printGrade();

public:
    AppX();
    ~AppX();
    int run();
};

AppX::~AppX()
{
    // You can use the traditional loop, which is more clear.
    for (vector<Class *>::iterator it = classVec.begin();
         it != classVec.end();
         ++it) {
        if (*it) delete (*it);
    }
    // You can use modern and simpler loops only if you know what it is doing.
    /*for (const auto &s: studentVec) {
        if (s) delete (s);
    }
    */
}

AppX::AppX()
{
    loadFiles();
}

void AppX::loadFiles()
{
    string line;
    size_t pos1, pos2;
    vector<string> bufv;
    Student *st = nullptr;
    string clsname;
    int point;
    Class *cl = nullptr;

    // Open a file as an input stream.
    ifstream stfile("./Students.txt");

    while (getline(stfile, line)) {
        if (line.empty()) // It's an empty line.
            continue;
        if (line[0] == '#') // It's a comment line.
            continue;

        // The bufv vector stores each column in the line.
        bufv.clear();
        // Split the line into columns.
        //   pos1: beginning of the column.
        //   pos2: end of the column.
        pos1 = 0;
        while (true) {
            pos2 = line.find(';', pos1 + 1);
            if (pos2 == string::npos) { // No more columns.
                // Save the last column (pos1 ~ eol).
                bufv.push_back(line.substr(pos1, string::npos));
                break;
            } else {
                // Save the column (pos1 ~ pos2).
                bufv.push_back(line.substr(pos1, pos2 - pos1));
            }
            pos1 = pos2 + 1;
        }

        // TODO: uncomment next lines after implementing class Undergraduate and Graduate.

        if (bufv[3] == "U")
            st = new Undergraduate(bufv[0], bufv[1], bufv[2]);
        else
            st = new Graduate(bufv[0], bufv[1], bufv[2]);


        studentVec.push_back(st);
    }
    stfile.close();

    // TODO: load data from ./Classes.txt and push objects to the vector.

    ifstream clfile("./Classes.txt");
    std::string temp_name;
    while(getline(clfile,line)){
        if(line.empty())continue;
        if(line[0]=='#')continue;
        if(line[0]=='C'){
            pos1=line.find(':');
            temp_name=line.substr(pos1+1,string::npos);
        }
        if(line[0]=='P'){
            point=line[7]-'0';
            cl=new Class(temp_name,point);
            classVec.push_back(cl);
        }
        if(line[0]=='F' || line[0]=='B'){
            for(int i=0;i<=studentVec.size()-1;i++){
                if(line==studentVec[i]->id){
                    cl->addStudent(*studentVec[i]);
                    studentVec[i]->addClass(cl);
                }
            }
        }
    }
    clfile.close();
}

void AppX::inputScore()
{
    // TODO: implement inputScore.
    // Hint: Take a look at printHighestScore().
    string sbuf;
    Class *cl=nullptr;
    while(true){
        cout<<"Please input the class name (or input q to quit): ";
        cin>> sbuf;
        if(sbuf == "q")break;
        cl=nullptr;
        for (vector<Class*>::iterator it = classVec.begin();
             it != classVec.end();
             ++ it) {
            if ((*it)->name == sbuf) {
                cl = *it;
                break;
            }
        }
        if (cl == nullptr) {
            cout << "No match class!\n";
            continue;
        }
        while(true){
            cout<<"Please input the student id and score(or input q to quit):";
            size_t pos;
            string line,id;
            double score;
            cin>>line;
            if(line=="q")break;
            pos=line.find(',');
            id=line.substr(0,pos);

            bool id_legal=false;
            for(int i=0;i<=studentVec.size()-1;i++){
                if(id==studentVec[i]->id)id_legal=true;
            }
            if(id_legal==false){cout<<"No match student!\n";continue;}
            else {
                score=stoi(line.substr(pos+1,string::npos));
                if(score<0 || score>100){cout<<"Wrong score!\n";continue;}
                cl->getStudentWrapper(id).setScore(score);
            }
            cout<<'\n';
        }
    }
}

void AppX::printHighestScore()
{
    string sbuf;
    Class *cl=nullptr;
    double highest;

    while (true) {
        cout << "Please input the class name (or input q to quit): ";
        cin >> sbuf;
        if (sbuf == "q")
            break;

        cl = nullptr;
        for (vector<Class*>::iterator it = classVec.begin();
                it != classVec.end();
                ++ it) {
            if ((*it)->name == sbuf) {
                cl = *it;
                break;
            }
        }
        if (cl == nullptr) {
            cout << "No match class!\n";
            continue;
        }

        highest = cl->getHighestScore();
        cout << setiosflags(ios::fixed) << setprecision(2)<< "The highest score is: " << highest << '\n';
    }
}

void AppX::printGrade() {
    // TODO: implement printGrade.
    // Hint: Take a look at printHighestScore().
    string sbuf;
    Student *stu;
    while (true) {
        stu=nullptr;
        cout << "Please input the student id (or input q to quit): ";
        cin >> sbuf;
        if (sbuf == "q")
            break;

        for (vector<Student *>::iterator it = studentVec.begin();
             it != studentVec.end();
             ++it) {
            if ((*it)->id == sbuf) {
                stu = *it;
                break;
            }
        }
        if (stu == nullptr) {
            cout << "No match student!\n";
            continue;
        }
        cout << stu->toString();
        cout << setiosflags(ios::fixed) << setprecision(2) << "GPA,AVG = " << stu->getGpa() << ',' << stu->getAvgScore() << '\n';
    }
}
    int AppX::run() {
        char cmd;
        while (true) {
            cout << "Command menu:\n"
                 << "\ti: Input score\n"
                 << "\ta: Compute highest score of a class\n"
                 << "\tg: Compute grade of a student\n"
                 << "\tq: Quit\n"
                 << "Please input the command: ";
            cin >> cmd;
            if (cmd == 'i') {
                inputScore();
            } else if (cmd == 'a') {
                printHighestScore();
            } else if (cmd == 'g') {
                printGrade();
            } else if (cmd == 'q') {
                break;
            } else {
                cout << "Invalid command!\n" << endl;
            }
        }
        return 0;
    }

    int main() {
        AppX app;
        return app.run();
    }
