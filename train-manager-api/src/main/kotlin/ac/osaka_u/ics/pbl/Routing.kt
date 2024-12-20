package ac.osaka_u.ics.pbl

import ac.osaka_u.ics.pbl.handler.AssignmentsHandler
import ac.osaka_u.ics.pbl.handler.ClientsHandler
import ac.osaka_u.ics.pbl.handler.QueueHandler
import ac.osaka_u.ics.pbl.handler.TasksHandler
import ac.osaka_u.ics.pbl.model.ClientRequest
import ac.osaka_u.ics.pbl.model.PostGeneratorRequest
import ac.osaka_u.ics.pbl.model.TaskRequest
import io.ktor.http.*
import io.ktor.resources.*
import io.ktor.server.application.*
import io.ktor.server.auth.*
import io.ktor.server.request.*
import io.ktor.server.resources.*
import io.ktor.server.resources.post
import io.ktor.server.response.*
import io.ktor.server.routing.*
import org.koin.ktor.ext.inject

@Resource("/tasks")
class TaskResource{
    @Resource("{id}")
    class TaskId(val parent: TaskResource = TaskResource(), val id: String)
    @Resource("generators")
    class Generators(val parent: TaskResource = TaskResource()){
        @Resource("{id}")
        class GeneratorId(val parent: Generators = Generators(), val id: Int)
    }
}

@Resource("/assignments")
class AssignmentsResource{
    @Resource("next")
    class Next(val parent: AssignmentsResource = AssignmentsResource())
    @Resource("{id}")
    class AssignmentId(val parent: AssignmentsResource = AssignmentsResource(), val id: String){
        @Resource("register")
        class Register(val parent: AssignmentId)
        @Resource("refresh")
        class Refresh(val parent: AssignmentId)
    }
}

@Resource("/queue")
class QueueResource

@Resource("/clients")
class ClientsResource

fun Application.configureRouting() {
    val assignmentsHandler by inject<AssignmentsHandler>()
    val queueHandler by inject<QueueHandler>()
    val tasksHandler by inject<TasksHandler>()
    val clientsHandler by inject<ClientsHandler>()
    routing {
        authenticate {
            get<AssignmentsResource.Next> {
                val clientId = call.principal<UserIdPrincipal>()!!.name
                val res = assignmentsHandler.handleGetNextAssignment(clientId)
                if (res != null) {
                    call.respond(res)
                } else {
                    call.respond(HttpStatusCode.NoContent)
                }
            }
            post<AssignmentsResource.AssignmentId.Register> { assignmentId ->
                val clientId = call.principal<UserIdPrincipal>()!!.name
                call.respond(assignmentsHandler.handleRegisterAssignment(assignmentId.parent.id, clientId))
            }
            post<AssignmentsResource.AssignmentId.Refresh> { assignmentId ->
                val clientId = call.principal<UserIdPrincipal>()!!.name
                call.respond(assignmentsHandler.handleRefreshAssignment(assignmentId.parent.id, clientId))
            }

            get<QueueResource> {
                call.respond(queueHandler.handleGetAll())
            }
            post<QueueResource> {
                val request = call.receive<TaskRequest>()
                call.respond(queueHandler.handlePost(request))
            }

            get<TaskResource> {
                call.respond(tasksHandler.handleGetTasks())
            }
            get<TaskResource.TaskId> { taskId ->
                call.respond(tasksHandler.handleGetTask(taskId.id))
            }
            get<TaskResource.Generators> {
                call.respond(tasksHandler.handleGetGenerators())
            }
            post<TaskResource.Generators> {
                val request = call.receive<PostGeneratorRequest>()
                val resp = tasksHandler.handlePostGenerator(request)
                call.respond(HttpStatusCode.Created, resp)
            }
            get<TaskResource.Generators.GeneratorId> { generatorId ->
                call.respond(tasksHandler.handleGetGenerator(generatorId.id))
            }
            delete<TaskResource.Generators.GeneratorId> { generatorId ->
                tasksHandler.handleDeleteGenerator(generatorId.id)
                call.respond(HttpStatusCode.NoContent)
            }
        }
        post<ClientsResource> {
            val request = call.receive<ClientRequest>()
            val clientId = clientsHandler.handleRegisterNewClient(request)
            call.respond(HttpStatusCode.Created, clientId)
        }
    }
}
