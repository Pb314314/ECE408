
#include	<wb.h>


wbArg_t wbArg_new(void) {
    wbArg_t arg;

    wb_init();

    wbArg_setInputCount(arg, 0);
    wbArg_setInputFiles(arg, NULL);
    wbArg_setOutputFile(arg, NULL);
    wbArg_setType(arg, NULL);
    wbArg_setExpectedOutputFile(arg, NULL);
    return arg;
}

void wbArg_delete(wbArg_t arg) {
    if (wbArg_getInputCount(arg) > 0 &&
            wbArg_getInputFiles(arg) != NULL) {
        int ii;
        for (ii = 0; ii < wbArg_getInputCount(arg); ii++) {
            wbDelete(wbArg_getInputFile(arg, ii));
        }
        wbDelete(wbArg_getInputFiles(arg));
        wbArg_setInputCount(arg, 0);
        wbArg_setInputFiles(arg, NULL);
    }
    if (wbArg_getOutputFile(arg)) {
        wbDelete(wbArg_getOutputFile(arg));
        wbArg_setOutputFile(arg, NULL);
    }
    if (wbArg_getExpectedOutputFile(arg)) {
        wbDelete(wbArg_getExpectedOutputFile(arg));
        wbArg_setExpectedOutputFile(arg, NULL);
    }
    if (wbArg_getType(arg)) {
        wbDelete(wbArg_getType(arg));
        wbArg_setType(arg, NULL);
    }
    return ;
}

// take char array as input return the number of files 
static int getInputFileCount(char * arg) {
    int count = 1;
    while (*arg != '\0' && *arg != '-') {
        if (*arg == ',') {      // use , to separate files
            count++;
        }
        arg++;
    }
    return count;
}

// parse the input file path and set value to resCount(number of files)
static char ** parseInputFiles(char * arg, int * resCount) {
    int count;
    int ii = 0;
    char ** files;      // pointer to an array of char array
    char * token;

    count = getInputFileCount(arg);

    files = wbNewArray(char *, count);

    token = strtok(arg, ",");       // separate input files using ,
    while (token != NULL) { 
        files[ii++] = wbString_duplicate(token);
        token = strtok(NULL, ",");
    }
    *resCount = ii;
    return files;
}

static char * parseString(char * arg) {
    return wbString_duplicate(arg);
}
// parse the input arg, set value of wbAtg_t and return it
wbArg_t wbArg_read(int argc, char ** argv) {
    int ii;
    wbArg_t arg;

    arg = wbArg_new();
    for (ii = 0; ii < argc; ii++) {
        if (wbString_startsWith(argv[ii], "-i")) {
            int fileCount;
            char ** files;      // a pointer point to an array of char array
            files = parseInputFiles(argv[ii + 1], &fileCount);

            wbArg_setInputCount(arg, fileCount);        // assign inputCount to arg
            wbArg_setInputFiles(arg, files);            // assign input file to arg
        } else if (wbString_startsWith(argv[ii], "-o")) {
            char * file = parseString(argv[ii + 1]);        // parse the set output file path
            wbArg_setOutputFile(arg, file);
        } else if (wbString_startsWith(argv[ii], "-e")) {
            char * file = parseString(argv[ii + 1]);        // parse the expected output file path
            wbArg_setExpectedOutputFile(arg, file);
        } else if (wbString_startsWith(argv[ii], "-t")) {
            char * type = parseString(argv[ii + 1]);        // parse the type
            wbArg_setType(arg, type);
        } else if (argv[ii][0] == '-') {
            wbLog(ERROR, "Unexpected program option ", argv[ii]);
        }
    }

    return arg;
}


