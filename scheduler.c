#define _XOPEN_SOURCE
#define _XOPEN_SOURCE_EXTENDED

#include "scheduler.h"

#include <assert.h>
#include <curses.h>
#include <ucontext.h>

#include "util.h"

// This is an upper limit on the number of tasks we can create.
#define MAX_TASKS 128

// This is the size of each task's stack memory
#define STACK_SIZE 65536

// task states
#define RUN 0
#define SLEEP 1
#define INPUT 2
#define READY 3
#define FINISH 4


// This struct will hold all the necessary information for each task
typedef struct task_info {
    // This field stores all the state required to switch back to this task
    ucontext_t context;

    // This field stores another context. This one is only used when the task
    // is exiting.
    ucontext_t exit_context;

    // TODO: Add fields here so you can:
    //   a. Keep track of this task's state.
    //   b. If the task is sleeping, when should it wake up?
    //   c. If the task is waiting for another task, which task is it waiting for?
    //   d. Was the task blocked waiting for user input? Once you successfully
    //      read input, you will need to save it here so it can be returned.

    // 0 for running; 1 for sleeping; 2 for waiting user input;
    // 3 for ready to run; 4 for finished
    // see macros defined above
    int state;
    //time this task should wake up; init to be 0
    size_t wake_time;
    //if the task is not waiting for any task, this is -1
    task_t waiting_task;
    //store user input; -1 for none
    int userinput;

} task_info_t;

int current_task = 0; //< The handle of the currently-executing task
int num_tasks = 1;    //< The number of tasks created so far
task_info_t tasks[MAX_TASKS]; //< Information for every task


/**
 * Initialize the scheduler. Programs should call this before calling any other
 * functiosn in this file.
 */
void scheduler_init() {
    //init fields for main task
    tasks[current_task].state = RUN;
    tasks[current_task].wake_time = 0;
    tasks[current_task].waiting_task= -1;
    tasks[current_task].userinput= -1;
}

//check whether task t is waiting for other task
bool is_waiting(task_t t) {
    task_t waiting_task = tasks[t].waiting_task;
    //if not waiting for any task
    if (waiting_task == -1) return false;
    //if the task it's waiting for has finished running
    if (tasks[waiting_task].state == FINISH) {
        tasks[t].waiting_task = -1;
        return false;
    }
    return true;
}

//select the index of the next task to be run
// return the task index
task_t scheduler_select_next() {
    while (true) {
        for (int i=0;i<num_tasks;i++) {
            //gets task after the current task
            task_t cur = (i+current_task+1)%num_tasks;
            //if the task is sleeping, check whether it has passed wake up time
            if (tasks[cur].state == SLEEP) {
                size_t cur_time = time_ms();
                if (cur_time >= tasks[cur].wake_time) tasks[cur].state = READY;
            }
            //if the task is waiting for userinput, try to get one
            if (tasks[cur].state == INPUT) {
                int c = getch();
                if (c != ERR) {
                    tasks[cur].state = READY;
                    tasks[cur].userinput = c;
                }
            }
            //if the task is ready to run, check whether it's waiting for others
            if (tasks[cur].state == READY) {
                if (is_waiting(cur)) continue;
                return cur;
            }
        }
    }
}

//run the task chosen by scheduler_select_next()
void scheduler_run_next() {
    task_t next = scheduler_select_next();
    task_t orig_task = current_task;
    current_task = next;
    tasks[current_task].state = RUN;
    //swap context to run the task
    if (swapcontext(&(tasks[orig_task].context), &(tasks[current_task].context))==-1) {
        perror("swap context failed!\n");
    }
}


/**
 * This function will execute when a task's function returns. This allows you
 * to update scheduler states and start another task. This function is run
 * because of how the contexts are set up in the task_create function.
 */
void task_exit() {
    // Handle the end of a task's execution here
    tasks[current_task].state = FINISH;
    scheduler_run_next();
}

/**
 * Create a new task and add it to the scheduler.
 *
 * \param handle  The handle for this task will be written to this location.
 * \param fn      The new task will run this function.
 */
void task_create(task_t* handle, task_fn_t fn) {
    // Claim an index for the new task
    int index = num_tasks;
    num_tasks++;

    // Set the task handle to this index, since task_t is just an int
    *handle = index;

    // We're going to make two contexts: one to run the task, and one that runs at the end of the task so we can clean up. Start with the second

    // First, duplicate the current context as a starting point
    getcontext(&tasks[index].exit_context);

    // Set up a stack for the exit context
    tasks[index].exit_context.uc_stack.ss_sp = malloc(STACK_SIZE);
    tasks[index].exit_context.uc_stack.ss_size = STACK_SIZE;

    // Set up a context to run when the task function returns. This should call task_exit.
    makecontext(&tasks[index].exit_context, task_exit, 0);

    // Now we start with the task's actual running context
    getcontext(&tasks[index].context);

    // Allocate a stack for the new task and add it to the context
    tasks[index].context.uc_stack.ss_sp = malloc(STACK_SIZE);
    tasks[index].context.uc_stack.ss_size = STACK_SIZE;

    // Now set the uc_link field, which sets things up so our task will go to the exit context when the task function finishes
    tasks[index].context.uc_link = &tasks[index].exit_context;

    // And finally, set up the context to execute the task function
    makecontext(&tasks[index].context, fn, 0);

    //init the fields for the new task
    tasks[index].state=READY;
    tasks[index].wake_time=0;
    tasks[index].waiting_task=-1;
    tasks[index].userinput=-1;
}

/**
 * Wait for a task to finish. If the task has not yet finished, the scheduler should
 * suspend this task and wake it up later when the task specified by handle has exited.
 *
 * \param handle  This is the handle produced by task_create
 */
void task_wait(task_t handle) {
    if (tasks[handle].state == FINISH) {
        return;
    }
    tasks[current_task].state = READY;
    //add the handle to the field of current task
    tasks[current_task].waiting_task = handle;
    scheduler_run_next();
}

/**
 * The currently-executing task should sleep for a specified time. If that time is larger
 * than zero, the scheduler should suspend this task and run a different task until at least
 * ms milliseconds have elapsed.
 * 
 * \param ms  The number of milliseconds the task should sleep.
 */
void task_sleep(size_t ms) {
    // Block this task until the requested time has elapsed.
    tasks[current_task].state = SLEEP;
    tasks[current_task].wake_time= ms+time_ms();
    scheduler_run_next();
}

/**
 * Read a character from user input. If no input is available, the task should
 * block until input becomes available. The scheduler should run a different
 * task while this task is blocked.
 *
 * \returns The read character code
 */
int task_readchar() {
    // Block this task until there is input available.
    // To check for input, call getch(). If it returns ERR, no input was available.
    // Otherwise, getch() will returns the character code that was read.
    if (tasks[current_task].userinput != -1) {
        int c = tasks[current_task].userinput;
        tasks[current_task].userinput = -1;
        return c;
    }
    tasks[current_task].state = INPUT;
    //let scheduler run something else first
    scheduler_run_next();
    //now it definitely has something valid in the userinput field
    int c = tasks[current_task].userinput;
    tasks[current_task].userinput = -1;
    return c;
}
