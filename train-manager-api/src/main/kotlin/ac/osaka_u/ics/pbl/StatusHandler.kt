package ac.osaka_u.ics.pbl

import ac.osaka_u.ics.pbl.model.ApiError
import io.ktor.http.*
import io.ktor.server.application.*
import io.ktor.server.plugins.statuspages.*
import io.ktor.server.response.*

fun Application.configureStatus() {
    install(StatusPages) {
        exception<Throwable> { call,cause ->
            cause.printStackTrace()
            if(cause is ApiException){
                when(cause){
                    is ApiException.NotFoundException -> call.respond(HttpStatusCode.NotFound, ApiError.NOT_FOUND)
                    is ApiException.BadRequestException -> call.respond(HttpStatusCode.BadRequest, ApiError(cause.message.orEmpty()))
                    is ApiException.UnauthorizedException -> call.respond(HttpStatusCode.Unauthorized, ApiError.UNAUTHORIZED)
                    is ApiException.ForbiddenException -> call.respond(HttpStatusCode.Forbidden, ApiError.FORBIDDEN)
                    is ApiException.InternalServerErrorException -> call.respond(HttpStatusCode.InternalServerError, ApiError(cause.message.orEmpty()))
                }
            }else{
                call.respond(HttpStatusCode.InternalServerError, ApiError("Unintended error occurred in the server"))
            }
        }
    }
}