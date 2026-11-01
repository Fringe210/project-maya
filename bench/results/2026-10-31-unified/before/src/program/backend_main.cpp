// One inference process; architecture implementations share the build, server protocol and GPU kernels.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
int maya_glm_main(int,char**);
int maya_ds4_main(int,char**);
int maya_mm3_main(int,char**);
int main(int argc,char** argv) {
    std::string family,pack;
    std::vector<char*> args; args.push_back(argv[0]);
    for(int i=1;i<argc;++i) {
        if(std::strcmp(argv[i],"--family")==0) {
            if(i+1==argc){std::fprintf(stderr,"--family needs deepseek, glm or minimax\n");return 2;}
            family=argv[++i];continue;
        }
        if((std::strcmp(argv[i],"--pack")==0 || std::strcmp(argv[i],"--glm-pack")==0) && i+1<argc) {
            pack=argv[i+1]; if(std::strcmp(argv[i],"--glm-pack")==0 && family.empty()) family="glm";
        }
        if(std::strcmp(argv[i],"--list-families")==0){std::puts("deepseek\nglm\nminimax");return 0;}
        args.push_back(argv[i]);
    }
    if(family.empty() && !pack.empty()) {
        auto p=std::filesystem::path(pack);
        if(std::filesystem::exists(p/"native_experts.txt")) family="glm";
        else if(p.filename()=="maya-ds4")family="deepseek";
        else if(p.filename()=="maya-mm3")family="minimax";
    }
    if(family.empty()) {
        std::puts("Maya unified backend: DeepSeek, GLM and MiniMax in one executable.\nUse --family deepseek|glm|minimax followed by the model options.\nUse --list-families to list supported architectures.");
        return argc==1 || (argc==2 && std::strcmp(argv[1],"--help")==0) ? 0 : 2;
    }
    args.push_back(nullptr);
    if(family=="deepseek" || family=="ds4")return maya_ds4_main((int)args.size()-1,args.data());
    if(family=="minimax" || family=="mm3") {
        // MiniMax's original IQ-only build did not use DS4's programmatic dependent launches.
#if defined(_WIN32)
        _putenv_s("DS4_PDL","0");
#else
        setenv("DS4_PDL","0",1);
#endif
        return maya_mm3_main((int)args.size()-1,args.data());
    }
    if(family=="glm")return maya_glm_main((int)args.size()-1,args.data());
    std::fprintf(stderr,"Unknown architecture: %s\n",family.c_str());return 2;
}
