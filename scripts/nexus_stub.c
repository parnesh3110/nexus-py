/*
 * Nexus.app's own executable. It runs Python *inside this process*, so macOS sees "Nexus"
 * everywhere: Camera/Microphone/Accessibility permissions, the Dock, the app switcher and
 * Activity Monitor. (A shell script that starts python3.11 made macOS credit everything to
 * "python3.11" instead.) The same approach py2app and PyInstaller use.
 *
 * Two builds (scripts/make_app.sh, scripts/build_release.sh):
 *   development  -DNEXUS_ROOT=... -DNEXUS_PYHOME=...: runs the project folder's code and .venv
 *   release      -DNEXUS_BUNDLED: everything is inside the app, found relative to this executable
 *                (Contents/Resources/python, site-packages, app, models), so the app works from
 *                any folder on any Mac
 *
 *   Started with no arguments (Finder, Dock, Spotlight): runs `python -m nexus.bootstrap`.
 *   Started with arguments: behaves like the python interpreter, so anything that re-runs
 *   sys.executable (a helper process, a restart) still works.
 */
#include <Python.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef NEXUS_BUNDLED
#include <mach-o/dyld.h>
#else
#ifndef NEXUS_ROOT
#error "NEXUS_ROOT must be defined (the nexus-py folder), or build with -DNEXUS_BUNDLED"
#endif
#ifndef NEXUS_PYHOME
#error "NEXUS_PYHOME must be defined (the base Python install behind .venv)"
#endif
#endif

static void show_dialog(const char *text) {
    char cmd[2048];
    snprintf(cmd, sizeof cmd,
             "osascript -e 'display dialog \"%s\" buttons {\"OK\"} default button 1 with title \"Nexus\"'", text);
    int shown = system(cmd);
    (void)shown;
}

static int launched_by_finder(int argc, char **argv) {
    /* Older macOS versions pass a -psn_... argument when an app is opened from Finder */
    return argc == 1 || (argc == 2 && strncmp(argv[1], "-psn", 4) == 0);
}

#ifdef NEXUS_BUNDLED
/* .../Nexus.app/Contents/MacOS/Nexus -> .../Nexus.app/Contents/Resources */
static int resources_dir(char *out, size_t n) {
    char exe[PATH_MAX], real[PATH_MAX];
    uint32_t size = sizeof exe;
    if (_NSGetExecutablePath(exe, &size) != 0 || realpath(exe, real) == NULL) return -1;
    char *slash = strrchr(real, '/');          /* strip "Nexus" */
    if (!slash) return -1;
    *slash = '\0';
    slash = strrchr(real, '/');                /* strip "MacOS" */
    if (!slash) return -1;
    *slash = '\0';
    return snprintf(out, n, "%s/Resources", real) < (int)n ? 0 : -1;
}

/* mkdir -p: create every missing folder along the way */
static int make_dirs(const char *path) {
    char tmp[PATH_MAX + 64];
    snprintf(tmp, sizeof tmp, "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    return (mkdir(tmp, 0755) == 0 || access(tmp, F_OK) == 0) ? 0 : -1;
}
#endif

int main(int argc, char **argv) {
    setenv("PYTHONUNBUFFERED", "1", 1);
    setenv("PATH", "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin", 1);

#ifdef NEXUS_BUNDLED
    char res[PATH_MAX + 16], home[PATH_MAX + 64], app[PATH_MAX + 64], support[PATH_MAX + 64];
    if (resources_dir(res, sizeof res) != 0) {
        show_dialog("Nexus could not find its own files. Please download it again.");
        return 1;
    }
    snprintf(home, sizeof home, "%s/python", res);
    snprintf(app, sizeof app, "%s/app", res);
    setenv("PYTHONHOME", home, 1);
    setenv("PYTHONPATH", app, 1);
    setenv("NEXUS_BUNDLED", "1", 1);
    setenv("NEXUS_RESOURCES", res, 1);
    /* A shipped app must never modify itself (it would break its code signature, and /Applications
       may not be writable): the build precompiles everything, so Python never writes .pyc files */
    setenv("PYTHONDONTWRITEBYTECODE", "1", 1);
    /* Your files live in Application Support; start there */
    const char *user_home = getenv("HOME");
    snprintf(support, sizeof support, "%s/Library/Application Support/Nexus", user_home ? user_home : "/tmp");
    if (make_dirs(support) != 0 || chdir(support) != 0) {
        show_dialog("Nexus could not open its Application Support folder.");
        return 1;
    }
#else
    setenv("PYTHONHOME", NEXUS_PYHOME, 1);
    setenv("PYTHONPATH", NEXUS_ROOT, 1);
    if (chdir(NEXUS_ROOT) != 0) {
        show_dialog("Nexus can not find its project folder (" NEXUS_ROOT
                    "). Was it moved? Run scripts/make_app.sh again from its new place.");
        return 1;
    }
#endif
    if (!launched_by_finder(argc, argv)) {
        return Py_BytesMain(argc, argv); /* act as the interpreter */
    }
    char *args[] = {argv[0], "-m", "nexus.bootstrap", NULL};
    return Py_BytesMain(3, args);
}
