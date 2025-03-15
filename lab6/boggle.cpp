#include "lexicon.h"
#include <iostream>
#include <vector>
using namespace std;

class BoggleGame{
public:
    int n;//棋盘边长
    vector<string> board;//棋盘
    vector<string> wordsFound1;//string类型向量 存放玩家1找到的单词们
    vector<string> wordsFound2;//string类型向量 存放玩家2找到的单词们
    set<string> autoFoundWords;//结算时，自动寻找的单词存放在此
    vector<vector<bool>> visited;//寻找单个单词dfs时的visited数组
    vector<vector<bool>> visited_printAllWords;//自动寻找所有可能单词时的dfs-visited数组
    Lexicon *lexicon;//Lexicon类，查找单词用
    int *score;//两个玩家的分数

    BoggleGame(int len);//BoggleGame类的构造函数
    void loadBoard();//读取输入内容，并加载棋盘
    void startGame();//开始游戏
    void playerTurn(int playerId);//在玩家的回合中进行的操作
    bool wordIsOnBoard(string word);//单词是否在棋盘上，包装函数
    void settleGame();//游戏结算
    bool hasBeenFound(string word,int playerId);//一个单词是否已经被找出过
    bool wordExist(string word);//单词是否在词典中
    void printAllPossibleWords();//打印出所有可能的答案
    string convertToLower(string word);//将任意输入/单词全部转为小写
    bool dfs(int row, int col, const string &word , int index,std::set<std::pair<int,int>>track);//DFS搜索，寻找单个单词用
    void dfs_PrintAllWords(int x, int y, std::string& currentWord);//自动寻找所有可能单词中的辅助函数dfs
    ~BoggleGame();
};
string BoggleGame::convertToLower(string word){//将单词转换为小写（已测试过）
    for(int i=0;i<word.size();i++){
        if(word[i]<='Z' && word[i]>='A')word[i]+='a'-'A';
    }
    return word;
}

BoggleGame::BoggleGame(int len){
    this->n =len;
    lexicon = new Lexicon ("EnglishWords.txt");
    score = new int [2];
    score[0]=0;
    score[1]=0;
    board.clear();
    wordsFound1.clear();
    wordsFound2.clear();
    visited.clear();
    visited_printAllWords.clear();
    autoFoundWords.clear();
}

void BoggleGame::settleGame(){
    cout<<"Player 1 Score: "<<score[0]<<endl;
    cout<<"Player 2 Score: "<<score[1]<<endl;
    if(score[0] > score[1])cout<<"Player 1 wins!"<<endl;
    else if(score[0] < score[1]) cout<<"Player 2 wins!"<<endl;
    else cout<<"It's a tie!"<<endl;
    printAllPossibleWords();
}

bool BoggleGame::hasBeenFound(string word,int playerId){
    bool found = false;
    if(playerId == 0){
        for(int i=0;i<wordsFound1.size();i++){
            if(wordsFound1[i] == convertToLower(word))found=true;
        }
    }
    else {
        for(int i=0;i<wordsFound2.size();i++){
            if(wordsFound2[i] == convertToLower(word))found=true;
        }
    }
    return found;
}

bool BoggleGame::wordExist(string word){//判断单词是否存在（在词典里）
    word = convertToLower(word);
    return lexicon->contains(word);
}

void BoggleGame::dfs_PrintAllWords(int x, int y, std::string& currentWord) {
    // 边界条件
    if (x < 0 || x >= this->n || y < 0 || y >= this->n || visited_printAllWords[x][y])return;
    // 更新当前的单词/部分单词
    currentWord += board[x][y];
    // 剪枝：如果当前前缀不在词典中，返回
    if (!lexicon->containsPrefix(currentWord)) {
        currentWord.pop_back(); // 回溯
        return;
    }
    // 如果当前单词是有效单词且至少有4个字，添加到结果中
    if (currentWord.size()>=4 && lexicon->contains(currentWord)) {
        autoFoundWords.insert(currentWord); // 使用set自动排序
    }

    // 标记当前字母为已访问
    visited_printAllWords[x][y] = true;

    // 深度优先搜索相邻的字母
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            if (dx != 0 || dy != 0) { // 跳过自身
                dfs_PrintAllWords(x + dx, y + dy, currentWord);
            }
        }
    }

    // 回溯
    visited_printAllWords[x][y] = false;
    currentWord.pop_back();
}




void BoggleGame::printAllPossibleWords(){
    cout<<"All Possible Words: ";
    for (int i = 0; i < board.size(); ++i) {
        for (int j = 0; j < board[0].size(); ++j) {
            std::string currentWord;
            dfs_PrintAllWords(i, j, currentWord);
        }
    }
    // 打印找到的单词，按字母顺序
    for (const auto& word : autoFoundWords) {
        cout << word << ' ';
    }
}


void BoggleGame::loadBoard(){
    string tmp;
    for (int i = 0; i < this->n; ++i) {
        getline(cin,tmp);
        board.push_back(tmp);
    }//把board初始化好

    visited.resize(n, vector<bool>(n, false));  // 将 visited 初始化为 n*n 的矩阵，每个元素都是 false
    visited_printAllWords.resize(n, vector<bool>(n, false));
}



void BoggleGame::startGame(){
    //showBoard();
    playerTurn(0);//玩家1先来
    playerTurn(1);//玩家2来
    settleGame();//结算游戏
}
void BoggleGame::playerTurn(int playerId){
    cout<<"Player "<<playerId+1<<" Score: "<<score[playerId]<<endl;
    string input;
    cin>>input;
    if(input =="???")return;
    else if(input.size()<=3){
        //过短
        cout<<input<<" is too short."<<endl;
    }
    else if(!wordExist(input)){
        //词典里没有
        cout<<input<<" is not a word."<<endl;
    }
    else if(!wordIsOnBoard(input)){
        //不在棋盘上
        cout<<input<<" is not on board."<<endl;
    }
    else if(hasBeenFound(input,playerId)){
        //已经说过的不能再说
        cout<<input<<" is already found."<<endl;
    }
    else{//单词符合要求
        if(playerId == 0)wordsFound1.push_back(convertToLower(input));//转为小写之后存入wordsFound1
        else wordsFound2.push_back(convertToLower(input));//转为小写之后存入wordsFound2
        score[playerId]+= input.size()-3;
        cout<<"Correct."<<endl;
    }
    playerTurn(playerId);
}

bool BoggleGame::wordIsOnBoard(string word){
    for(int i=0;i<this->n;i++){
        for(int j=0;j<this->n;j++){
            //cout<<"start find "<<word[0]<<" in ("<<i<<','<<j<<')'<<endl;
            if(board[i][j] == word[0] || board[i][j] == word[0]+'a'-'A' || board[i][j] == word[0]+'A'-'a'){//对每个可能的起始位置进行查找
                //cout<<"find "<<word[0]<<" in ("<<i<<','<<j<<')'<<endl;
                std::set<std::pair<int,int>>track;
                track.clear();
                if(dfs(i,j,word,0,track))return true;
            }
        }
    }
    return false;
}

bool BoggleGame::dfs(int row,int col,const string &word, int index,std::set<std::pair<int,int>>track){
    //cout<<endl<<"start find "<<word;
    if(index == word.size()-1){//最后一步，终止条件
        if(board[row][col] != word[index]
           && board[row][col]+'a'-'A' != word[index]
           && board[row][col]+'A'-'a' != word[index])return false;
        else return true;
    }
    else{//还在嵌套中
        if(board[row][col] != word[index]
           && board[row][col]+'a'-'A' != word[index]
           && board[row][col]+'A'-'a' != word[index])return false;
        else{//这个位置上的字母符合
            //cout<<"found "<<word[index]<<" in ("<<row<<','<<col<<')'<<endl;
            //将该点添加至track中
            std::set<std::pair<int,int>>tmp = track;
            tmp.insert(std::pair<int,int>(row,col));
            //去寻找下一个字母所在的非track中的位置
            int index_row;
            int index_col;
            bool result=false;
            /*
            index_row = (row==0)?(n-1):(row-1);
            for(int i=0;i<3;i++) {
                index_col = (col==0)?(n-1):(col-1);
                for (int j = 0; j < 3; j++) {
                    //如果想搜索的点已经在track中则continue
                    auto it = tmp.find({index_row, index_col});
                    if (it != tmp.end()) {//下一个想寻找的点在track中
                        index_col = (index_col == n - 1) ? (0) : (index_col + 1);
                        continue;
                    }
                    else if (board[index_row][index_col] != word[index + 1]
                               && board[index_row][index_col] + 'a' - 'A' != word[index + 1]
                               && board[index_row][index_col] + 'A' - 'a' != word[index + 1]) {//字母不对，显然不行，不可添加track
                        index_col = (index_col == n - 1) ? (0) : (index_col + 1);
                        continue;
                    }
                    result = result || dfs(index_row, index_col, word, index + 1, tmp);
                    index_col = (index_col == n - 1) ? (0) : (index_col + 1);
                }
                index_row = (index_row == n - 1) ? (0) : (index_row + 1);
            }
            */

            //检查周围八个邻居
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    // 跳过当前位置 (0, 0)
                    if (dr == 0 && dc == 0) continue;

                    index_row = row + dr;
                    index_col = col + dc;

                    // 检查是否超出边界
                    if (index_row < 0 || index_row >= n || index_col < 0 || index_col >= n) {
                        continue; // 如果超出边界，则跳过该位置
                    }

                    // 如果要搜索的点已经在 track 中，跳过
                    if (tmp.find({index_row, index_col}) != tmp.end()) {
                        continue;
                    }

                    // 检查下一个字母是否匹配
                    char nextChar = board[index_row][index_col];
                    if (std::tolower(nextChar) != std::tolower(word[index + 1])) {
                        continue; // 如果字母不匹配，则跳过
                    }

                    // 递归调用 dfs 并更新结果
                    result = result || dfs(index_row, index_col, word, index + 1, tmp);

                    // 如果已经找到结果，可以提前返回
                    if (result) return true;
                }
            }

            return result;
        }
    }
}
BoggleGame::~BoggleGame(){
    delete score;
    delete lexicon;
}



int main() {
    // TODO
    int len;
    cin>>len;//输入棋盘规模
    cin.ignore(); // 忽略换行符
    BoggleGame game(len);//创建游戏
    game.loadBoard();//输入棋盘
    game.startGame();//开始游戏（进入player1的回合，再player2的回合，再结算游戏）
    return 0;
}
