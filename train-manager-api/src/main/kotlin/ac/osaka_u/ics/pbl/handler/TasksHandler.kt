package ac.osaka_u.ics.pbl.handler

import ac.osaka_u.ics.pbl.domain.model.Task
import ac.osaka_u.ics.pbl.domain.model.TaskGenerator
import ac.osaka_u.ics.pbl.domain.repos.TaskRepository
import ac.osaka_u.ics.pbl.model.PostGeneratorRequest

class TasksHandler(taskRepo: TaskRepository, generatorRepo: TaskGeneratorRepository) {
    fun handleGetTasks(): List<Task> {

    }

    fun handleGetTask(id: String): Task {

    }

    fun handleGetGenerators(): List<TaskGenerator> {

    }

    fun handleGetGenerator(id: Int): TaskGenerator {

    }

    fun handlePostGenerator(request: PostGeneratorRequest): TaskGenerator {

    }

    fun handleDeleteGenerator(id: Int) {

    }
}