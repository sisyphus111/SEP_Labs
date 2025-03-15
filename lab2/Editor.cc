#include <iostream>
#include <sstream>
#include "Editor.h"

using namespace std;

Editor::Editor()
{
    buffer = new Buffer();
}
Editor::~Editor()//正确与否存疑
{
    // TODO: Implement destructor
    delete buffer;
}

void Editor::run()
{
    string cmd;
    while (true)
    {
        cout << "cmd> ";
        cout.flush();
        getline(cin, cmd);
        if (cmd == "Q")
            break;
        try {
            dispatchCmd(cmd);
        } catch (const char *e) {
            cout << "? " << e << endl;
        } catch (const out_of_range &oor) {
            cout << "? " << oor.what() << endl;
        } catch (const range_error &re) {
            cout << "? " << re.what() << endl;
        }
    }
}
void Editor::cmdAppend()
{
    cout << "It's input mode now. Quit with a line with a single dot(.)" << endl;
    // TODO: finish cmdAppend.
    while (true){
        string text;
        getline(cin, text);
        if (text == ".")break;
        else buffer->appendLine(text);
    }
}
void Editor::cmdSwap(int a, int b){
    buffer->swap(a,b);
}
void Editor::cmdInsert()
{
    cout << "It's input mode now. Quit with a line with a single dot(.)" << endl;
    bool firstLine = true;
    while (true)
    {
        string text;
        getline(cin, text);
        if (text == ".")
            break;
        else {
            if (firstLine) {
            buffer->insertLine(text);
            firstLine = false;
            }
            else {buffer->appendLine(text);}
        }
    }
}

void Editor::cmdDelete(int start, int end)
{
    buffer->deleteLines(start, end);
}

void Editor::cmdNull(int line)
{
    cout << buffer->moveToLine(line) << endl;
}

void Editor::cmdNumber(int start, int end)
{
    buffer->showLines(start, end);
}

void Editor::cmdWrite(const string &filename)
{
    buffer->writeToFile(filename);
}

void Editor::dispatchCmd(const string &cmd)
{
    if (cmd == "a") {
        cmdAppend();
        return;
    }
    if (cmd == "i") {
        cmdInsert();
        return;
    }
    if (cmd[0] == 'w' && cmd[1] == ' ') {
        // TODO: call cmdWrite with proper arguments
        string name_file=cmd.substr(2);
        cmdWrite(name_file);
        return;
    }
    // TODO: handle special case "1,$n".
    int start, end;
    char comma, type;
    if(cmd[0]<'0' || cmd[0]>'9')throw "Bad/Unknown command";
    stringstream ss(cmd);
    ss >> start;
    if (ss.eof()) {
        cmdNull(start);
        return;
    }//若无逗号且合法，在上面都已经处理完毕
    size_t pos = cmd.find(',');
    if (pos == std::string::npos) throw "Bad/Unknown command";
    string endstr;
    ss>>comma>>endstr;
    if(endstr=="$n" && start==1){cmdNumber(1,buffer->length);return;}
    else if(endstr=="$n" && start!=1){throw "Bad/Unknown command";}
    else{
        type=endstr.back();
        endstr.pop_back();//从endstr里截出最后一位作为type
        stringstream endss(endstr);
        endss>>end;//endstr里剩下的放入end
        if(!endss.eof())throw "Bad/Unknown command";
    }

    /*ss >> comma >> end >> type;*/
    /*if (ss.good()) {*/
        if (type == 'n') {
            cmdNumber(start, end);
            return;
        } else if (type == 'd') {
            cmdDelete(start, end);
            return;
        } else if (type == 's') {
            cmdSwap(start,end);
            return;
        }  else if (type == 'l') {
            // for test, please use ?,?l -> ?,?s -> ?,?t
            buffer->loadAddr2(start+1, end+1);
            // buffer->printAddr(start);
            // buffer->printAddr(end);
            return;
        } else if (type == 't') {
            buffer->testSwap(start+1, end+1);
            // buffer->printAddr(start);
            // buffer->printAddr(end);
            return;
        }
    /*}*/
    throw "Bad/Unknown command";
}
